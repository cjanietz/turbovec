package turbovec

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

const testDim = 8

var errShape = errors.New("unexpected search shape")

func row(dim int, seed float32) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = seed + float32(i)*0.1
	}
	return v
}

func basis(dim, lo, hi int) []float32 {
	v := make([]float32, dim)
	for i := lo; i < hi && i < dim; i++ {
		v[i] = 1
	}
	return v
}

func concat(rows ...[]float32) []float32 {
	var out []float32
	for _, r := range rows {
		out = append(out, r...)
	}
	return out
}

func TestConstructRejectsBadBitWidth(t *testing.T) {
	_, err := NewTurboQuantIndex(testDim, 5)
	if err == nil || !strings.Contains(err.Error(), "bit_width") {
		t.Fatalf("expected bit_width error, got %v", err)
	}
}

func TestConstructRejectsBadDim(t *testing.T) {
	_, err := NewTurboQuantIndex(7, 4)
	if err == nil || !strings.Contains(err.Error(), "multiple of 8") {
		t.Fatalf("expected dim error, got %v", err)
	}
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func TestLazyDimAndAdd(t *testing.T) {
	idx, err := NewLazyTurboQuantIndex(4)
	if err != nil {
		t.Fatal(err)
	}
	if idx.Dim() != nil {
		t.Fatalf("lazy dim want nil, got %v", idx.Dim())
	}
	if idx.Len() != 0 {
		t.Fatalf("len = %d", idx.Len())
	}
	if err := idx.Add(concat(row(testDim, 1), row(testDim, 2)), testDim); err != nil {
		must(t, err)
	}
	if got := idx.Dim(); got == nil || *got != testDim {
		t.Fatalf("dim = %v", got)
	}
	if idx.Len() != 2 {
		t.Fatalf("len = %d", idx.Len())
	}
}

func TestAddDimMismatch(t *testing.T) {
	idx, err := NewTurboQuantIndex(testDim, 4)
	if err != nil {
		must(t, err)
	}
	err = idx.Add(row(16, 1), 16)
	if err == nil || !strings.Contains(err.Error(), "dim mismatch") {
		t.Fatalf("expected dim mismatch, got %v", err)
	}
}

func TestAddRejectsNonFinite(t *testing.T) {
	idx, err := NewTurboQuantIndex(testDim, 4)
	if err != nil {
		must(t, err)
	}
	v := row(testDim, 1)
	v[3] = float32(math.NaN())
	err = idx.Add(v, testDim)
	if err == nil || !strings.Contains(err.Error(), "invalid input value") {
		t.Fatalf("expected invalid input, got %v", err)
	}
}

func TestSearchFindsSelf(t *testing.T) {
	idx, err := NewTurboQuantIndex(testDim, 4)
	if err != nil {
		must(t, err)
	}
	a, b := basis(testDim, 0, testDim/2), basis(testDim, testDim/2, testDim)
	if err := idx.Add(concat(a, b), testDim); err != nil {
		must(t, err)
	}
	res, err := idx.Search(a, testDim, 1)
	if err != nil {
		must(t, err)
	}
	if res.NQ != 1 || res.K != 1 {
		t.Fatalf("shape nq=%d k=%d", res.NQ, res.K)
	}
	if res.IndicesForQuery(0)[0] != 0 {
		t.Fatalf("top-1 = %v, want slot 0", res.IndicesForQuery(0))
	}
}

func TestSearchWithMask(t *testing.T) {
	idx, err := NewTurboQuantIndex(testDim, 4)
	if err != nil {
		must(t, err)
	}
	a, b := basis(testDim, 0, testDim/2), basis(testDim, testDim/2, testDim)
	if err := idx.Add(concat(a, b), testDim); err != nil {
		must(t, err)
	}
	mask := []bool{false, true}
	res, err := idx.SearchWithMask(a, testDim, 2, mask)
	if err != nil {
		must(t, err)
	}
	if res.K != 1 || res.IndicesForQuery(0)[0] != 1 {
		t.Fatalf("masked result = %+v", res)
	}
}

func TestSwapRemove(t *testing.T) {
	idx, err := NewTurboQuantIndex(testDim, 4)
	if err != nil {
		must(t, err)
	}
	if err := idx.Add(concat(row(testDim, 1), row(testDim, 2), row(testDim, 3)), testDim); err != nil {
		must(t, err)
	}
	moved, err := idx.SwapRemove(0)
	if err != nil {
		must(t, err)
	}
	if moved != 2 {
		t.Fatalf("moved from %d, want 2", moved)
	}
	if idx.Len() != 2 {
		t.Fatalf("len = %d", idx.Len())
	}
}

func TestBytesRoundTrip(t *testing.T) {
	idx, err := NewTurboQuantIndex(testDim, 4)
	if err != nil {
		must(t, err)
	}
	if err := idx.Add(row(testDim, 1.5), testDim); err != nil {
		must(t, err)
	}
	loaded, err := TurboQuantIndexFromBytes(idx.ToBytes())
	if err != nil {
		must(t, err)
	}
	if loaded.Len() != 1 || loaded.BitWidth() != 4 {
		t.Fatalf("loaded len=%d bw=%d", loaded.Len(), loaded.BitWidth())
	}
}

func TestWriteLoadAndSync(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "index.tv")
	idx, err := NewTurboQuantIndex(testDim, 4)
	if err != nil {
		must(t, err)
	}
	if err := idx.Add(row(testDim, 1), testDim); err != nil {
		must(t, err)
	}
	if err := idx.Write(path); err != nil {
		must(t, err)
	}
	loaded, err := LoadTurboQuantIndex(path)
	if err != nil {
		must(t, err)
	}
	if loaded.Len() != 1 {
		t.Fatalf("load len=%d", loaded.Len())
	}
	if err := loaded.Add(row(testDim, 2), testDim); err != nil {
		must(t, err)
	}
	if err := loaded.Sync(path); err != nil {
		must(t, err)
	}
	again, err := LoadTurboQuantIndex(path)
	if err != nil {
		must(t, err)
	}
	if again.Len() != 2 {
		t.Fatalf("sync load len=%d", again.Len())
	}
	if _, err := os.Stat(path); err != nil {
		must(t, err)
	}
}

func TestCalibrate(t *testing.T) {
	idx, err := NewTurboQuantIndex(testDim, 4)
	if err != nil {
		must(t, err)
	}
	if idx.CalibrationState() != "uncalibrated" {
		t.Fatalf("state = %s", idx.CalibrationState())
	}
	sample := concat(row(testDim, 1), row(testDim, 20), row(testDim, -3), row(testDim, 8))
	if err := idx.Calibrate(sample, testDim); err != nil {
		must(t, err)
	}
	if idx.CalibrationState() != "calibrated" {
		t.Fatalf("state = %s", idx.CalibrationState())
	}
}

func TestConcurrentSearch(t *testing.T) {
	idx, err := NewTurboQuantIndex(testDim, 4)
	if err != nil {
		must(t, err)
	}
	var batch []float32
	for i := 0; i < 32; i++ {
		batch = append(batch, row(testDim, float32(i))...)
	}
	if err := idx.Add(batch, testDim); err != nil {
		must(t, err)
	}
	idx.Prepare()
	q := row(testDim, 3)
	var wg sync.WaitGroup
	errCh := make(chan error, 8)
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				res, err := idx.Search(q, testDim, 3)
				if err != nil {
					errCh <- err
					return
				}
				if res.K != 3 || len(res.Indices) != 3 {
					errCh <- errShape
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			must(t, err)
		}
	}
}

func TestIdMapSearchRemoveContains(t *testing.T) {
	idx, err := NewIdMapIndex(testDim, 4)
	if err != nil {
		must(t, err)
	}
	ids := []uint64{1001, 1002, 1003}
	if err := idx.AddWithIDs(concat(basis(testDim, 0, 4), basis(testDim, 4, 8), basis(testDim, 2, 6)), testDim, ids); err != nil {
		must(t, err)
	}
	if !idx.Contains(1002) || idx.Contains(9) {
		t.Fatalf("contains failed")
	}
	res, err := idx.Search(basis(testDim, 0, 4), testDim, 1)
	if err != nil {
		must(t, err)
	}
	if res.IDsForQuery(0)[0] != 1001 {
		t.Fatalf("top id = %v", res.IDsForQuery(0))
	}
	allow := []uint64{1003}
	res, err = idx.SearchWithAllowlist(row(testDim, 1), testDim, 5, allow)
	if err != nil {
		must(t, err)
	}
	if res.K != 1 || res.IDsForQuery(0)[0] != 1003 {
		t.Fatalf("allowlist = %+v", res)
	}
	if !idx.Remove(1002) || idx.Contains(1002) {
		t.Fatalf("remove failed")
	}
	if idx.Len() != 2 {
		t.Fatalf("len = %d", idx.Len())
	}
}

func TestIdMapBytesRoundTrip(t *testing.T) {
	idx, err := NewIdMapIndex(testDim, 4)
	if err != nil {
		must(t, err)
	}
	if err := idx.AddWithIDs(row(testDim, 1), testDim, []uint64{7}); err != nil {
		must(t, err)
	}
	loaded, err := IdMapIndexFromBytes(idx.ToBytes())
	if err != nil {
		must(t, err)
	}
	if !loaded.Contains(7) {
		t.Fatal("id 7 missing after from_bytes")
	}
}

func TestIdMapWriteLoad(t *testing.T) {
	path := filepath.Join(t.TempDir(), "index.tvim")
	idx, err := NewIdMapIndex(testDim, 4)
	if err != nil {
		must(t, err)
	}
	if err := idx.AddWithIDs(row(testDim, 1), testDim, []uint64{42}); err != nil {
		must(t, err)
	}
	if err := idx.Write(path); err != nil {
		must(t, err)
	}
	loaded, err := LoadIdMapIndex(path)
	if err != nil {
		must(t, err)
	}
	if !loaded.Contains(42) {
		t.Fatal("id 42 missing after load")
	}
}
