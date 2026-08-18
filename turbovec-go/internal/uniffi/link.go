package uniffi

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo linux LDFLAGS: -L${SRCDIR}/../../../target/release -Wl,-rpath,${SRCDIR}/../../../target/release -lturbovec_go -ldl -lm -lpthread
#cgo darwin LDFLAGS: -L${SRCDIR}/../../../target/release -Wl,-rpath,${SRCDIR}/../../../target/release -lturbovec_go
*/
import "C"
