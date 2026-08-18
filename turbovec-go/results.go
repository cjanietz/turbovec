package turbovec

// SearchResults is a batch of positional top-k hits, row-major nq × k.
type SearchResults struct {
	Scores  []float32
	Indices []int64
	NQ      int
	K       int
}

// ScoresForQuery returns the scores for query row qi.
func (r SearchResults) ScoresForQuery(qi int) []float32 {
	if r.K == 0 {
		return nil
	}
	start := qi * r.K
	return r.Scores[start : start+r.K]
}

// IndicesForQuery returns the slot indices for query row qi.
func (r SearchResults) IndicesForQuery(qi int) []int64 {
	if r.K == 0 {
		return nil
	}
	start := qi * r.K
	return r.Indices[start : start+r.K]
}

// IDSearchResults is a batch of id-mapped top-k hits, row-major nq × k.
type IDSearchResults struct {
	Scores []float32
	IDs    []uint64
	NQ     int
	K      int
}

// ScoresForQuery returns the scores for query row qi.
func (r IDSearchResults) ScoresForQuery(qi int) []float32 {
	if r.K == 0 {
		return nil
	}
	start := qi * r.K
	return r.Scores[start : start+r.K]
}

// IDsForQuery returns the external ids for query row qi.
func (r IDSearchResults) IDsForQuery(qi int) []uint64 {
	if r.K == 0 {
		return nil
	}
	start := qi * r.K
	return r.IDs[start : start+r.K]
}
