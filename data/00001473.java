public class TapException extends CascadingException { Tuple payload; public TapException() { } public TapException( String string ) { super( string ); } public TapException( String string, Throwable throwable ) { super( string, throwable ); } public TapException( String string, Throwable throwable, Tuple payload ) { super( string, throwable ); this.payload = payload; } public TapException( String string, Tuple payload ) { super( string ); this.payload = payload; } public TapException( Throwable throwable ) { super( throwable ); } public TapException( Tap tap, Fields incomingFields, Fields selectorFields, Throwable throwable ) { super( createMessage( tap, incomingFields, selectorFields ), throwable ); } public Tuple getPayload() { return payload; } private static String createMessage( Tap tap, Fields incomingFields, Fields selectorFields ) { String message = "unable to resolve scheme sink selector: " + selectorFields.printVerbose() + ", with incoming: " + incomingFields.printVerbose(); return TraceUtil.formatTrace( tap.getScheme(), message ); } }