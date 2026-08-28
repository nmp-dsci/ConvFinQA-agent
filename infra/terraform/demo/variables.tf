# These defaults ARE the live configuration. There is no committed tfvars file:
# a second source of truth for a single-environment stack is a way to deploy the
# wrong thing, not a way to stay flexible. Override on the command line for a
# one-off.

variable "region" {
  description = "AWS region. Same as the sibling demos, so they share the tfstate bucket."
  type        = string
  default     = "ap-southeast-2"
}

variable "project" {
  description = "Name prefix for every resource in this stack."
  type        = string
  default     = "convfinqa"
}
