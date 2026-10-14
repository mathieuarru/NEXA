#!/usr/bin/env bash
# Download PrimeKG v2.1 to primekg/data/kg.csv and verify its MD5.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
DESTINATION="${DATA_DIR}/kg.csv"
PARTIAL="${DESTINATION}.part"

PRIMEKG_URL="https://dataverse.harvard.edu/api/access/datafile/6180620"
PRIMEKG_MD5="aac8191d4fbc5bf09cdf8c3c78b4e75f"

mkdir -p "${DATA_DIR}"

if [[ -f "${DESTINATION}" ]] && echo "${PRIMEKG_MD5}  ${DESTINATION}" | md5sum --check --status -; then
  echo "PrimeKG is already downloaded and its checksum is valid: ${DESTINATION}"
  exit 0
fi

echo "Downloading PrimeKG v2.1 (Harvard Dataverse file 6180620)..."
curl --location --fail --show-error \
  --retry 5 --retry-delay 5 --continue-at - \
  --output "${PARTIAL}" "${PRIMEKG_URL}"

echo "${PRIMEKG_MD5}  ${PARTIAL}" | md5sum --check --strict -
mv "${PARTIAL}" "${DESTINATION}"
echo "PrimeKG downloaded and verified: ${DESTINATION}"
