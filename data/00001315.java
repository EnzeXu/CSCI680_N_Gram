public class FlowTapException extends FlowException { public FlowTapException ( ) { } public FlowTapException ( String message ) { super ( message ) ; } public FlowTapException ( String message , Throwable throwable ) { super ( message , throwable ) ; } public FlowTapException ( String flowName , String message , Throwable throwable ) { super ( flowName , message , throwable ) ; } public FlowTapException ( String flowName , String message ) { super ( flowName , message ) ; } public FlowTapException ( Throwable throwable ) { super ( throwable ) ; } }