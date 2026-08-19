package turbovec

import (
	"github.com/cjanietz/turbovec/turbovec-go/internal/uniffi"
)

// WarningHandler receives non-fatal diagnostics (for example a durable
// save whose directory fsync failed after the rename had already
// committed). It may be called from any thread.
type WarningHandler = uniffi.WarningHandler

// SetWarningHandler installs a process-wide warning sink. Passing nil
// restores turbovec's stderr default.
func SetWarningHandler(handler WarningHandler) {
	if handler == nil {
		uniffi.SetWarningHandler(nil)
		return
	}
	uniffi.SetWarningHandler(&handler)
}
