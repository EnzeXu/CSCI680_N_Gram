public class OperationException extends CascadingException { public OperationException ( ) { } public OperationException ( String string ) { super ( string ) ; } public OperationException ( String string , Throwable throwable ) { super ( string , throwable ) ; } public OperationException ( Throwable throwable ) { super ( throwable ) ; } }