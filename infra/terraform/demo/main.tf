# The demo stack: one ECR repo, one App Runner service. No VPC, no database,
# no Secrets Manager — the container is read-only by construction (DEMO_MODE is
# baked into the image) and holds no keys.
#
# State lives in the account's existing tfstate bucket under this project's own
# key, so nothing new has to be created or secured. First-time order (locally,
# or let CI do it):
#
#   terraform init
#   terraform apply -target=aws_ecr_repository.demo   # repo before first push
#   ../../../scripts/aws_build_push.sh                 # image must exist first
#   terraform apply                                    # then the service starts

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
  backend "s3" {
    bucket       = "data-qa-tfstate-089783391188"
    key          = "convfinqa-agent/demo.tfstate"
    region       = "ap-southeast-2"
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}

# ── Image registry ───────────────────────────────────────────────────────

resource "aws_ecr_repository" "demo" {
  name                 = "${var.project}-demo"
  image_tag_mutability = "MUTABLE" # :latest is the deploy pointer

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "demo" {
  repository = aws_ecr_repository.demo.name
  # The image bakes in the dataset and the committed eval artifacts, so old
  # versions are not small. Keep a short rollback window, not an archive.
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep the 5 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}

# ── App Runner ───────────────────────────────────────────────────────────

data "aws_iam_policy_document" "apprunner_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["build.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "apprunner_ecr_access" {
  name               = "${var.project}-apprunner-ecr-access"
  description        = "Lets App Runner pull the ${var.project} demo image from ECR."
  assume_role_policy = data.aws_iam_policy_document.apprunner_trust.json
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr_access" {
  role       = aws_iam_role.apprunner_ecr_access.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

resource "aws_apprunner_service" "demo" {
  service_name = "${var.project}-demo"

  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_ecr_access.arn
    }
    # Pushing :latest redeploys — the deploy workflow's build+push IS the
    # release step; `terraform apply` after it only reconciles drift.
    auto_deployments_enabled = true
    image_repository {
      image_identifier      = "${aws_ecr_repository.demo.repository_url}:latest"
      image_repository_type = "ECR"
      image_configuration {
        port = "8080"
        # Nothing to configure and no secrets to reference, deliberately:
        # DEMO_MODE is baked into the image so infrastructure cannot turn the
        # public deployment into a billable one.
      }
    }
  }

  instance_configuration {
    # Measured, not guessed: 225 MiB RSS at rest with the dataset and demo pack
    # baked in, rising to 315 MiB once every committed prediction CSV is in the
    # read cache — which is what the answers explorer does on first visit, and
    # is the high-water mark for this service. 512 MB would survive at rest and
    # then OOM the first time someone opened that tab, so 1 GB is the floor.
    cpu    = "512"
    memory = "1024"
  }

  health_check_configuration {
    protocol            = "HTTP"
    path                = "/healthz"
    interval            = 10
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 5
  }

  observability_configuration {
    observability_enabled = false
  }
}

# One alarm: sustained 5xx means the demo is broken for whoever just opened the
# portfolio link — the one failure mode worth being told about.
resource "aws_cloudwatch_metric_alarm" "http_5xx" {
  alarm_name          = "${var.project}-demo-5xx"
  alarm_description   = "Sustained 5xx from the ${var.project} demo."
  namespace           = "AWS/AppRunner"
  metric_name         = "5xxStatusResponses"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 2
  threshold           = 10
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions = {
    ServiceName = aws_apprunner_service.demo.service_name
    ServiceID   = aws_apprunner_service.demo.service_id
  }
}

output "service_url" {
  description = "Public URL of the demo."
  value       = "https://${aws_apprunner_service.demo.service_url}"
}

output "ecr_repository_url" {
  description = "ECR repository the deploy workflow pushes to."
  value       = aws_ecr_repository.demo.repository_url
}
