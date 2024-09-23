public class OperationStatus { private final boolean isSuccess ; private final String message ; private final StatusCode statusCode ; public OperationStatus ( boolean success , String msg ) { this ( success , msg , null ) ; } public OperationStatus ( boolean success , String msg , StatusCode code ) { isSuccess = success ; message = msg ; statusCode = code ; } public boolean isSuccess ( ) { return isSuccess ; } public String getMessage ( ) { return message ; } public StatusCode getStatusCode ( ) { return statusCode ; } @ Override public String toString ( ) { return " { OperationStatus success=" + isSuccess + " : " + message + " } " ; } }