# These defaults ARE the live configuration. There is no committed tfvars file:
# a second source of truth for a single-environment stack is a way to deploy the
# wrong thing, not a way to stay flexible. Override on the command line for a
# one-off.

variable "region" {
  description = <<-EOT
    Region the demo runs in.

    Singapore rather than Sydney, for a reason worth recording: AWS applies a
    two-App-Runner-services-per-region restriction to this account, and Sydney
    is already at it (yt-agent-demo, data-qa-backend-api). The restriction is
    genuinely per-region — verified by creating and deleting a probe service —
    so a neighbouring region has a fresh allowance. Costs ~100ms of latency for
    an Australian visitor, which a replay-backed demo does not notice.

    The tfstate bucket stays in ap-southeast-2; the S3 backend region is
    independent of the provider region and does not move with this.
  EOT
  type        = string
  default     = "ap-southeast-1"
}

variable "project" {
  description = "Name prefix for every resource in this stack."
  type        = string
  default     = "convfinqa"
}
