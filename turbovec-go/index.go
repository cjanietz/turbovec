package turbovec

import "github.com/cjanietz/turbovec/turbovec-go/internal/uniffi"

// TurboQuantIndex is a positional TurboQuant index. Each vector is
// identified by its insertion slot; swap-remove invalidates later slots.
type TurboQuantIndex struct {
	inner *uniffi.TurboQuantIndex
}

// NewTurboQuantIndex constructs an index with a committed dim.
func NewTurboQuantIndex(dim, bitWidth int) (*TurboQuantIndex, error) {
	d, err := asU64("dim", dim)
	if err != nil {
		return nil, err
	}
	bw, err := asU64("bit_width", bitWidth)
	if err != nil {
		return nil, err
	}
	inner, err := uniffi.NewTurboQuantIndex(d, bw)
	if err != nil {
		return nil, mapErr(err)
	}
	return &TurboQuantIndex{inner: inner}, nil
}

// NewLazyTurboQuantIndex constructs an empty index whose dim is locked
// on the first Add.
func NewLazyTurboQuantIndex(bitWidth int) (*TurboQuantIndex, error) {
	bw, err := asU64("bit_width", bitWidth)
	if err != nil {
		return nil, err
	}
	inner, err := uniffi.TurboQuantIndexNewLazy(bw)
	if err != nil {
		return nil, mapErr(err)
	}
	return &TurboQuantIndex{inner: inner}, nil
}

// LoadTurboQuantIndex reads a .tv file written by Write or Sync.
func LoadTurboQuantIndex(path string) (*TurboQuantIndex, error) {
	inner, err := uniffi.TurboQuantIndexLoad(path)
	if err != nil {
		return nil, mapErr(err)
	}
	return &TurboQuantIndex{inner: inner}, nil
}

// TurboQuantIndexFromBytes deserializes an in-memory .tv image.
func TurboQuantIndexFromBytes(data []byte) (*TurboQuantIndex, error) {
	inner, err := uniffi.TurboQuantIndexFromBytes(data)
	if err != nil {
		return nil, mapErr(err)
	}
	return &TurboQuantIndex{inner: inner}, nil
}

// Add appends a flat row-major float32 batch of shape (n, dim).
func (idx *TurboQuantIndex) Add(vectors []float32, dim int) error {
	d, err := asU64("dim", dim)
	if err != nil {
		return err
	}
	return mapErr(idx.inner.Add(float32Bytes(vectors), d))
}

// Search returns the top-k slots for each query row.
func (idx *TurboQuantIndex) Search(queries []float32, dim, k int) (SearchResults, error) {
	return idx.SearchWithMask(queries, dim, k, nil)
}

// SearchWithMask is Search restricted to slots where mask[i] is true.
// A nil mask searches every slot; a non-nil mask must have length Len().
func (idx *TurboQuantIndex) SearchWithMask(queries []float32, dim, k int, mask []bool) (SearchResults, error) {
	d, err := asU64("dim", dim)
	if err != nil {
		return SearchResults{}, err
	}
	kk, err := asU64("k", k)
	if err != nil {
		return SearchResults{}, err
	}
	var maskArg *[]bool
	if mask != nil {
		maskArg = &mask
	}
	raw, err := idx.inner.SearchWithMask(float32Bytes(queries), d, kk, maskArg)
	if err != nil {
		return SearchResults{}, mapErr(err)
	}
	return SearchResults{
		Scores:  raw.Scores,
		Indices: raw.Indices,
		NQ:      int(raw.Nq),
		K:       int(raw.K),
	}, nil
}

// Calibrate fits TQ+ from a representative sample of shape (n, dim).
func (idx *TurboQuantIndex) Calibrate(sample []float32, dim int) error {
	d, err := asU64("dim", dim)
	if err != nil {
		return err
	}
	return mapErr(idx.inner.Calibrate(float32Bytes(sample), d))
}

// SwapRemove deletes slot i in O(1) by moving the last vector into it.
func (idx *TurboQuantIndex) SwapRemove(i int) (int, error) {
	u, err := asU64("idx", i)
	if err != nil {
		return 0, err
	}
	moved, err := idx.inner.SwapRemove(u)
	if err != nil {
		return 0, mapErr(err)
	}
	return int(moved), nil
}

// Prepare warms search caches so the first Search does not pay them.
func (idx *TurboQuantIndex) Prepare() {
	idx.inner.Prepare()
}

// Write persists a durable .tv snapshot.
func (idx *TurboQuantIndex) Write(path string) error {
	return mapErr(idx.inner.Write(path))
}

// WriteWithDurability persists a .tv snapshot. durable=false skips fsync.
func (idx *TurboQuantIndex) WriteWithDurability(path string, durable bool) error {
	return mapErr(idx.inner.WriteWithDurability(path, durable))
}

// Sync incrementally persists changes to path.
func (idx *TurboQuantIndex) Sync(path string) error {
	return mapErr(idx.inner.Sync(path))
}

// ToBytes serializes a .tv image in memory.
func (idx *TurboQuantIndex) ToBytes() []byte {
	return idx.inner.ToBytes()
}

// Len is the number of stored vectors.
func (idx *TurboQuantIndex) Len() int {
	return int(idx.inner.Len())
}

// Dim is the committed width, or nil on a lazy index that has not seen Add.
func (idx *TurboQuantIndex) Dim() *int {
	return optU64ToInt(idx.inner.Dim())
}

// BitWidth is 2, 3 or 4, fixed at construction.
func (idx *TurboQuantIndex) BitWidth() int {
	return int(idx.inner.BitWidth())
}

// CalibrationState is "uncalibrated" or "calibrated".
func (idx *TurboQuantIndex) CalibrationState() string {
	return idx.inner.CalibrationState()
}
