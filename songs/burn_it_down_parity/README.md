# Historical Burn It Down parity fixture

The riff inputs are copied from `baseline01_riff`. Subsequent step inputs are
extracted verbatim from the historical FLAC workflows recorded in
`reference_workflows.json`, which includes source file SHA-256 hashes. Runtime
request normalization strips surrounding ABC whitespace, as the reference does.
Explicit style/lyrics files are consumed verbatim, including trailing newlines.

`score_file`, `lyrics_file`, and `style_file` select files relative to this song
directory for a particular step. Other steps inherit the song-level files.

The 280-second continuation durations are generation limits, not expected output
lengths: EOS may finish earlier. Carried lengths and blend settings reproduce the
recorded chain. All steps consume the preceding custom-generated take.

Full-chain numerical parity is not yet established; see `docs/PARITY_PROGRESS.md`.
