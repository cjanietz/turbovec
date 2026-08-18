package turbovec

import "github.com/RyanCodrai/turbovec/turbovec-go/internal/uniffi"

// IdMapIndex is a TurboQuantIndex with stable external uint64 ids.
type IdMapIndex struct {
	inner *uniffi.IdMapIndex
}

// NewIdMapIndex constructs an id-mapped index with a committed dim.
func NewIdMapIndex(dim, bitWidth int) (*IdMapIndex, error) {
	d, err := asU64("dim", dim)
	if err != nil {
		return nil, err
	}
	bw, err := asU64("bit_width", bitWidth)
	if err != nil {
		return nil, err
	}
	inner, err := uniffi.NewIdMapIndex(d, bw)
	if err != nil {
		return nil, mapErr(err)
	}
	return &IdMapIndex{inner: inner}, nil
}

// NewLazyIdMapIndex constructs an empty index whose dim is locked on the
// first AddWithIDs.
func NewLazyIdMapIndex(bitWidth int) (*IdMapIndex, error) {
	bw, err := asU64("bit_width", bitWidth)
	if err != nil {
		return nil, err
	}
	inner, err := uniffi.IdMapIndexNewLazy(bw)
	if err != nil {
		return nil, mapErr(err)
	}
	return &IdMapIndex{inner: inner}, nil
}

// LoadIdMapIndex reads a .tvim file written by Write or Sync.
func LoadIdMapIndex(path string) (*IdMapIndex, error) {
	inner, err := uniffi.IdMapIndexLoad(path)
	if err != nil {
		return nil, mapErr(err)
	}
	return &IdMapIndex{inner: inner}, nil
}

// IdMapIndexFromBytes deserializes an in-memory .tvim image.
func IdMapIndexFromBytes(data []byte) (*IdMapIndex, error) {
	inner, err := uniffi.IdMapIndexFromBytes(data)
	if err != nil {
		return nil, mapErr(err)
	}
	return &IdMapIndex{inner: inner}, nil
}

// AddWithIDs appends vectors with the given external ids.
func (idx *IdMapIndex) AddWithIDs(vectors []float32, dim int, ids []uint64) error {
	d, err := asU64("dim", dim)
	if err != nil {
		return err
	}
	return mapErr(idx.inner.AddWithIds(float32Bytes(vectors), d, ids))
}

// Remove deletes id if present. Returns whether it was found.
func (idx *IdMapIndex) Remove(id uint64) bool {
	return idx.inner.Remove(id)
}

// Contains reports whether id is in the index.
func (idx *IdMapIndex) Contains(id uint64) bool {
	return idx.inner.Contains(id)
}

// Search returns the top-k external ids for each query row.
func (idx *IdMapIndex) Search(queries []float32, dim, k int) (IDSearchResults, error) {
	return idx.SearchWithAllowlist(queries, dim, k, nil)
}

// SearchWithAllowlist is Search restricted to the given ids. A nil
// allowlist searches every id; a non-nil allowlist must be non-empty
// and contain only ids currently in the index.
func (idx *IdMapIndex) SearchWithAllowlist(queries []float32, dim, k int, allowlist []uint64) (IDSearchResults, error) {
	d, err := asU64("dim", dim)
	if err != nil {
		return IDSearchResults{}, err
	}
	kk, err := asU64("k", k)
	if err != nil {
		return IDSearchResults{}, err
	}
	var allow *[]uint64
	if allowlist != nil {
		allow = &allowlist
	}
	raw, err := idx.inner.SearchWithAllowlist(float32Bytes(queries), d, kk, allow)
	if err != nil {
		return IDSearchResults{}, mapErr(err)
	}
	return IDSearchResults{
		Scores: raw.Scores,
		IDs:    raw.Ids,
		NQ:     int(raw.Nq),
		K:      int(raw.K),
	}, nil
}

// Calibrate fits TQ+ from a representative sample of shape (n, dim).
func (idx *IdMapIndex) Calibrate(sample []float32, dim int) error {
	d, err := asU64("dim", dim)
	if err != nil {
		return err
	}
	return mapErr(idx.inner.Calibrate(float32Bytes(sample), d))
}

// Prepare warms search caches and the id→slot map.
func (idx *IdMapIndex) Prepare() {
	idx.inner.Prepare()
}

// Write persists a durable .tvim snapshot.
func (idx *IdMapIndex) Write(path string) error {
	return mapErr(idx.inner.Write(path))
}

// WriteWithDurability persists a .tvim snapshot. durable=false skips fsync.
func (idx *IdMapIndex) WriteWithDurability(path string, durable bool) error {
	return mapErr(idx.inner.WriteWithDurability(path, durable))
}

// Sync incrementally persists changes to path.
func (idx *IdMapIndex) Sync(path string) error {
	return mapErr(idx.inner.Sync(path))
}

// ToBytes serializes a .tvim image in memory.
func (idx *IdMapIndex) ToBytes() []byte {
	return idx.inner.ToBytes()
}

// Len is the number of stored vectors.
func (idx *IdMapIndex) Len() int {
	return int(idx.inner.Len())
}

// Dim is the committed width, or nil on a lazy index that has not seen AddWithIDs.
func (idx *IdMapIndex) Dim() *int {
	return optU64ToInt(idx.inner.Dim())
}

// BitWidth is 2, 3 or 4, fixed at construction.
func (idx *IdMapIndex) BitWidth() int {
	return int(idx.inner.BitWidth())
}

// CalibrationState is "uncalibrated" or "calibrated".
func (idx *IdMapIndex) CalibrationState() string {
	return idx.inner.CalibrationState()
}
