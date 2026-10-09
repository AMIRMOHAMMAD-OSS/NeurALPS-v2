# NeurALPS interactive explorer

The Python/HTML interface is included in this update. Two data products remain
on Jean Zay and must be exported once: the existing fitted natural-reference
tables and the natural candidate feature bank. This is packaging, not training
or model inference.

## What each number means

| Display | Comparison | Scope |
| --- | --- | --- |
| Natural-reference percentile | Matched natural objects scored in their original assemblies | Original full-context single-object protocol, minimum 100 connected reference components |
| Natural-repertoire rank | Unique matching natural fragments scored against the same masked-query predictions | Selected contiguous segment, selected context mode, same types/topology |
| Activity model score | Sigmoid output of the all-494 supervised head | Whole assembly; not a calibrated probability |

The reference run is `e8_reference_scale_652486`, with 537 groups. Its scale was
fitted on a calibration partition of the 3,515 natural internal-test assemblies,
with a separate audit partition. Natural train/dev were used for overlap checks,
not scale fitting. The separate donor bank exports all **34,927** natural
assemblies: 27,913 train, 3,499 dev and 3,515 internal test.

## Publish the code update from Windows

The supplied update ZIP contains `neuralps-explorer.patch` and
`Apply-NeurALPS-Update.ps1`. Extract it and execute the script in PowerShell.
It clones the existing repository into a new Downloads folder, checks the patch,
creates a normal commit and pushes to `main`. It preserves the repository's
history and stops on any failed Git command. It does not upload or fabricate
the missing natural assets.

## Export once on Jean Zay

After the code update is on GitHub, run this in a Jean Zay terminal:

```bash
set -e
root=/lustre/fswork/projects/rech/nef/unh87ms/NeurALPSv2
release=/lustre/fswork/projects/rech/nef/unh87ms/NeurALPS_Explorer_20261010
out=/lustre/fswork/projects/rech/nef/unh87ms/NeurALPS_release_exports/NeurALPS_natural_20261010

test ! -e "$release"
test ! -e "$out"
git clone --depth 1 https://github.com/AMIRMOHAMMAD-OSS/NeurALPS-v2.git "$release"
"$root/envs/esmc_connections_hf/bin/python" "$release/scripts/export_natural_assets.py" \
  --root "$root" --output "$out"
```

If compute nodes/login nodes cannot clone GitHub, transfer the exporter kit
`NeurALPS_Explorer_Export_Kit_20261010.zip` from the update package through your
existing Mac/gateway route. Extract it into `$release`'s parent; it has the
enclosing directory `NeurALPS_Explorer_20261010`. Run the same final Python
command above. The kit contains only the source needed by the exporter.

The exporter checks the original reference hashes and source feature matrix,
reads the project databases in read-only mode, and copies the exact float32
cached embeddings. It makes no updates to the original project.

The reference asset is small. The full repertoire can total several GB; it is
split into roughly 92 MB uncompressed vector shards, individually below
GitHub's asset limit. Transfer only `*.zip` and `UPLOAD_MANIFEST.json` from the
output folder, not the uncompressed `repertoire` subdirectory.

To enable percentiles first, run the exporter with `--reference-only` and a
different output folder. Candidate search still requires the full export.

## Transfer the exported assets to Windows

Use your existing transfer route to copy only the ZIP assets and
`UPLOAD_MANIFEST.json` into
`C:\Users\YOUR_USER\Downloads\NeurALPS_natural_20261010`.
Do not copy the uncompressed `repertoire` subdirectory.

## Upload the natural data release from Windows

```powershell
& {
    $ErrorActionPreference = "Stop"
    $repo = "AMIRMOHAMMAD-OSS/NeurALPS-v2"
    $tag = "v0.2.0-20261010"
    $folder = Join-Path $env:USERPROFILE "Downloads\NeurALPS_natural_20261010"
    $manifest = Get-Content -Raw (Join-Path $folder "UPLOAD_MANIFEST.json") | ConvertFrom-Json

    foreach ($entry in $manifest.PSObject.Properties) {
        $file = Join-Path $folder $entry.Name
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $file).Hash.ToLowerInvariant() -ne $entry.Value.sha256) {
            throw "Checksum mismatch: $file"
        }
    }

    gh release view $tag --repo $repo *> $null
    if ($LASTEXITCODE -ne 0) {
        gh release create $tag --repo $repo --target main --title "NeurALPS interactive explorer" --notes "Original natural-reference scale and exact all-natural candidate embeddings. Explorer validation is documented in the repository." --prerelease
        if ($LASTEXITCODE -ne 0) { throw "Release creation failed." }
    }
    $uploaded = gh release view $tag --repo $repo --json assets | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw "Could not inspect release assets." }
    foreach ($entry in $manifest.PSObject.Properties) {
        $existing = @($uploaded.assets | Where-Object { $_.name -eq $entry.Name })
        if ($existing.Count) {
            Write-Host "Already present; preserving existing asset: $($entry.Name)"
            continue
        }
        gh release upload $tag (Join-Path $folder $entry.Name) --repo $repo
        if ($LASTEXITCODE -ne 0) { throw "Upload failed for $($entry.Name). Rerun to resume." }
    }
    Write-Host "Natural assets uploaded. Restart Colab and run the updated notebook."
}
```

If an existing release asset has the wrong contents, resolve that mismatch
explicitly; the commands above do not replace published files silently.

## Use the explorer

Open `notebooks/NeurALPS.ipynb` from `main` in a fresh Colab runtime. The source,
original runtime and reference scale download automatically. Select an A100 or
L4 for exact BF16 ESMC inference. The donor index and vector shards download on
the first search and are cached for subsequent searches in that session.

- Click one part to inspect its percentile, coordinates and reference support.
- Shift-click to select a contiguous region. Ctrl/Cmd-click can make an
  arbitrary selection for scoring; retrieval requires a contiguous region.
- “Include boundaries” adds boundaries touching selected domains explicitly.
- “Score selection together” uses one joint mask, including overlap and alias
  closure. It does not average previously computed individual percentiles.
- “Find natural alternatives” reports the exact count of higher, equal and
  lower compatible unique fragments under the declared ranking tolerance.
- “FASTA” exports individual physical feature sequences with their roles.
- “Variant JSON” splices a complete-domain span within one protein and keeps
  recipient flanks. “Test variant” rebuilds boundary windows and reruns ESMC,
  the frozen encoder and the supervised head before comparing outputs.
- “Save interactive HTML” preserves inspection and already computed results.
  Computing new scores or candidates requires the live Colab Python runtime.

## Remaining validation

The exporter must run against the actual Jean Zay database and the resulting
assets must be published before new-input percentiles and full-corpus retrieval
can be checked end to end. The portable encoder is checked against the original
BODE1, BODE2 and T-domain golden fixtures. Candidate math, duplicate handling,
alias exclusion, joint masking, splicing and the browser controls are checked
locally. New ESMC inference still uses the historical embedding gate in Colab.
