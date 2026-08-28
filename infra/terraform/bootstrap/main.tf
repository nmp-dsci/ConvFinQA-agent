# Bootstrap: run ONCE, locally, with admin credentials (aws sso login).
#
# Creates only the CI deploy role. Local state on purpose — this module is what
# the demo stack's remote state depends on, so it cannot itself live there.
#
#   cd infra/terraform/bootstrap
#   terraform init && terraform apply
#
# The GitHub OIDC *provider* already exists in this account (created by
# data-qa-agent's bootstrap) and is referenced, not recreated; two providers for
# the same issuer is an error. Deleting data-qa would take it with it — recreate
# it there, or move the resource here, if that ever happens.
#
# There is no demo-data bucket here, unlike the siblings: everything this demo
# serves is committed to the repo, so CI's checkout is the whole build context.

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  description = "Region the demo stack runs in. See ecr_regions for why this is not the only one that matters."
  type        = string
  default     = "ap-southeast-1"
}

variable "ecr_regions" {
  description = <<-EOT
    Regions the deploy role may manage ECR repositories in.

    ECR is regional and IAM ARNs embed the region, so a single-region grant
    silently locks CI out the moment the demo moves. It lists every region the
    stack has lived in rather than just the current one, so a rollback does not
    need a bootstrap re-apply with admin credentials to succeed.
  EOT
  type        = list(string)
  default     = ["ap-southeast-1", "ap-southeast-2"]
}

variable "project" {
  type    = string
  default = "convfinqa"
}

variable "github_repo" {
  description = "GitHub repo allowed to assume the deploy role."
  type        = string
  default     = "nmp-dsci/ConvFinQA-agent"
}

variable "tfstate_bucket" {
  description = "Existing state bucket shared with the sibling demos."
  type        = string
  default     = "data-qa-tfstate-089783391188"
}

data "aws_caller_identity" "current" {}

data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}

# ── CI deploy role (GitHub OIDC, no stored keys) ─────────────────────────

data "aws_iam_policy_document" "github_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    # This repo only — any branch or workflow ref within it, so a
    # workflow_dispatch from a branch under review still works.
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:*"]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  name               = "${var.project}-github-deploy"
  description        = "Assumed by GitHub Actions (OIDC) to deploy the ${var.project} demo."
  assume_role_policy = data.aws_iam_policy_document.github_trust.json
}

# Scoped to this project's name prefix wherever the service supports it.
data "aws_iam_policy_document" "deploy" {
  statement {
    sid = "TerraformStateKey"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
    ]
    resources = [
      "arn:aws:s3:::${var.tfstate_bucket}",
      "arn:aws:s3:::${var.tfstate_bucket}/${var.project}-agent/*",
    ]
  }

  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"] # This action does not support resource scoping.
  }

  statement {
    sid     = "EcrRepo"
    actions = ["ecr:*"]
    resources = [
      for r in var.ecr_regions :
      "arn:aws:ecr:${r}:${data.aws_caller_identity.current.account_id}:repository/${var.project}-*"
    ]
  }

  statement {
    sid       = "AppRunner"
    actions   = ["apprunner:*"]
    resources = ["*"] # App Runner's describe/list calls are not resource-scopable.
  }

  statement {
    sid = "IamForServiceRoles"
    actions = [
      "iam:GetRole",
      "iam:CreateRole",
      "iam:DeleteRole",
      "iam:TagRole",
      "iam:PassRole",
      "iam:ListRolePolicies",
      "iam:ListAttachedRolePolicies",
      "iam:ListInstanceProfilesForRole",
      "iam:AttachRolePolicy",
      "iam:DetachRolePolicy",
      "iam:PutRolePolicy",
      "iam:DeleteRolePolicy",
      "iam:GetRolePolicy",
      "iam:CreateServiceLinkedRole",
    ]
    resources = [
      "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.project}-*",
      "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/aws-service-role/*",
    ]
  }

  statement {
    sid = "Observability"
    actions = [
      "cloudwatch:PutMetricAlarm",
      "cloudwatch:DeleteAlarms",
      "cloudwatch:DescribeAlarms",
      "cloudwatch:ListTagsForResource",
      "cloudwatch:TagResource",
      "logs:CreateLogGroup",
      "logs:DescribeLogGroups",
      "logs:PutRetentionPolicy",
      "logs:ListTagsForResource",
      "logs:TagResource",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "deploy" {
  name   = "${var.project}-deploy"
  role   = aws_iam_role.github_deploy.id
  policy = data.aws_iam_policy_document.deploy.json
}

output "deploy_role_arn" {
  description = "Set this as DEPLOY_ROLE_ARN in .github/workflows/deploy-aws.yml."
  value       = aws_iam_role.github_deploy.arn
}
