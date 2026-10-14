#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
DESTINATION="${DATA_DIR}/BioHopR.json"
PARTIAL="${DESTINATION}.part"

# Pinned revision of the dataset and SHA-256 of BioHopR.json (its Git LFS object id).
BIOHOPR_REPO="knowlab-research/BioHopR"
BIOHOPR_COMMIT="08f06692c3900347e7405ba55ff6a6d330f55ddb"
BIOHOPR_SHA256="1adda2f23d305c394537e7b56d3e972b2d8f4141ae716ede54e0d08876eb527d"
BASE_URL="https://huggingface.co/datasets/${BIOHOPR_REPO}/resolve/${BIOHOPR_COMMIT}"

mkdir -p "${DATA_DIR}"

if [[ -f "${DESTINATION}" ]] && echo "${BIOHOPR_SHA256}  ${DESTINATION}" | sha256sum --check --status -; then
  echo "BioHopR is already downloaded and its checksum is valid: ${DESTINATION}"
  exit 0
fi

echo "Downloading BioHopR (${BIOHOPR_REPO} @ ${BIOHOPR_COMMIT:0:8})..."
curl --location --fail --show-error \
  --retry 5 --retry-delay 5 \
  --output "${PARTIAL}" "${BASE_URL}/BioHopR.json"

echo "${BIOHOPR_SHA256}  ${PARTIAL}" | sha256sum --check --strict -
mv "${PARTIAL}" "${DESTINATION}"

# Keep the dataset card next to the data.
curl --location --fail --show-error --retry 5 --retry-delay 5 \
  --output "${DATA_DIR}/README.md" "${BASE_URL}/README.md"

echo "BioHopR downloaded and verified: ${DESTINATION}"
