public class DocumentServiceException extends CascadingException { public DocumentServiceException() { } public DocumentServiceException( String string ) { super( string ); } public DocumentServiceException( String string, Throwable throwable ) { super( string, throwable ); } public DocumentServiceException( Throwable throwable ) { super( throwable ); } }