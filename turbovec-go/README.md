# turbovec-go

Go bindings for [turbovec](https://github.com/RyanCodrai/turbovec). The public
package is `turbovec`; import it as:

```go
import "github.com/RyanCodrai/turbovec/turbovec-go"
```

64-bit hosts only (`amd64` / `arm64`), matching the core crate.

## Build

The module talks to a native library built from the `turbovec-go` crate.

```bash
# from the repository root
cargo build -p turbovec-go --release
cd turbovec-go
CGO_ENABLED=1 go test
```

cgo is linked against `target/release/libturbovec_go` via an rpath in
`internal/uniffi/link.go`. Rebuild that crate after any Rust change.

To regenerate the committed UniFFI scaffolding (needs
[`uniffi-bindgen-go`](https://github.com/NordSecurity/uniffi-bindgen-go)
v0.7.1+v0.31.0):

```bash
./scripts/generate-go-bindings.sh
```

## API

Vectors are a flat `[]float32` of length `n * dim` — the same layout as
Rust `add_2d` / `try_search`.

```go
idx, err := turbovec.NewTurboQuantIndex(1536, 4)
if err != nil {
    log.Fatal(err)
}
if err := idx.Add(vectors, 1536); err != nil {
    log.Fatal(err)
}
res, err := idx.Search(query, 1536, 10)
if err != nil {
    log.Fatal(err)
}
_ = res.IndicesForQuery(0)

idx2, err := turbovec.NewIdMapIndex(1536, 4)
if err != nil {
    log.Fatal(err)
}
if err := idx2.AddWithIDs(vectors, 1536, []uint64{1001, 1002}); err != nil {
    log.Fatal(err)
}
ids, err := idx2.Search(query, 1536, 10)
_ = ids.IDsForQuery(0)
```

`Search` may run concurrently. `Add`, `Calibrate`, `SwapRemove`, `Sync`,
and `Remove` take a write lock.

`from_parts` and `turbovec::convert` stay Rust-only, as in the Python
binding.
