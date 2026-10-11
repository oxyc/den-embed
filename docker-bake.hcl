group "default" {
  targets = ["image"]
}

target "image" {
  context    = "."
  dockerfile = "Dockerfile"
  platforms  = ["linux/amd64"]
  pull       = true
  cache-from = ["type=gha,scope=release-images"]
  attest = [
    "type=provenance,mode=max",
    "type=sbom",
  ]
}
