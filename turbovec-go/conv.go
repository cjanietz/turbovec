package turbovec

import (
	"errors"
	"strings"
	"unsafe"
)

// MaxDim is the largest committed vector width the index accepts.
const MaxDim = 16384

func asU64(name string, n int) (uint64, error) {
	if n < 0 {
		return 0, errors.New(name + " must be non-negative")
	}
	return uint64(n), nil
}

func optU64ToInt(p *uint64) *int {
	if p == nil {
		return nil
	}
	v := int(*p)
	return &v
}

// float32Bytes reinterprets a float32 slice as little-endian bytes with
// no copy. All shipping turbovec targets are little-endian.
func float32Bytes(v []float32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

func mapErr(err error) error {
	if err == nil {
		return nil
	}
	s := err.Error()
	const prefix = "IndexError: Message: "
	if strings.HasPrefix(s, prefix) {
		return errors.New(strings.TrimPrefix(s, prefix))
	}
	return err
}
