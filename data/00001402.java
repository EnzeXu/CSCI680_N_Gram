public class TraceUtil { private static final Map<String, Pattern> registeredApiBoundaries = new ConcurrentHashMap<String, Pattern>(); private static interface TraceFormatter { String format( String trace ); } public static void registerApiBoundary( String apiBoundary ) { registeredApiBoundaries.put( apiBoundary, Pattern.compile( apiBoundary ) ); } public static void unregisterApiBoundary( String apiBoundary ) { registeredApiBoundaries.remove( apiBoundary ); } public static void setTrace( Object object, String trace ) { Util.setInstanceFieldIfExists( object, "trace", trace ); } private static String formatTrace( Traceable traceable, String message, TraceFormatter formatter ) { if( traceable == null ) return message; String trace = traceable.getTrace(); if( trace == null ) return message; return formatter.format( trace ) + " " + message; } public static String formatTraces( Collection<FlowElement> flowElements, String delim ) { List<String> messages = new ArrayList<>( flowElements.size() ); for( FlowElement flowElement : flowElements ) messages.add( formatTrace( (Traceable) flowElement, flowElement.toString(), new TraceFormatter() { @Override public String format( String trace ) { return "[" + trace + "] ->"; } } ) ); return Util.join( messages, delim ); } public static String formatTrace( final Scheme scheme, String message ) { return formatTrace( scheme, message, new TraceFormatter() { @Override public String format( String trace ) { return "[" + Util.truncate( scheme.toString(), 25 ) + "][" + trace + "]"; } } ); } public static String formatTrace( FlowElement flowElement, String message ) { if( flowElement == null ) return message; if( flowElement instanceof Pipe ) return formatTrace( (Pipe) flowElement, message ); if( flowElement instanceof Tap ) return formatTrace( (Tap) flowElement, message ); throw new UnsupportedOperationException( "cannot format type: " + flowElement.getClass().getName() ); } public static String formatRawTrace( Pipe pipe, String message ) { return formatTrace( pipe, message, new TraceFormatter() { @Override public String format( String trace ) { return "[" + trace + "]"; } } ); } public static String formatTrace( final Pipe pipe, String message ) { return formatTrace( pipe, message, new TraceFormatter() { @Override public String format( String trace ) { return "[" + Util.truncate( pipe.getName(), 25 ) + "][" + trace + "]"; } } ); } public static String formatTrace( final Tap tap, String message ) { return formatTrace( tap, message, new TraceFormatter() { @Override public String format( String trace ) { return "[" + Util.truncate( tap.toString(), 25 ) + "][" + trace + "]"; } } ); } public static String formatTrace( Operation operation, String message ) { if( !( operation instanceof BaseOperation ) ) return message; String trace = ( (BaseOperation) operation ).getTrace(); if( trace == null ) return message; return "[" + trace + "] " + message; } public static String captureDebugTrace( Object target ) { StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace(); StackTraceElement candidateUserCodeElement = null; StackTraceElement apiCallElement = null; Class<?> tracingBoundary = target.getClass(); String boundaryClassName = tracingBoundary.getName(); for( int i = stackTrace.length - 1; i >= 0; i-- ) { StackTraceElement stackTraceElement = stackTrace[ i ]; String stackClassName = stackTraceElement.getClassName(); boolean atApiBoundary = atApiBoundary( stackTraceElement.toString() ); if( ( stackClassName != null && ( stackClassName.startsWith( boundaryClassName ) ) || atApiBoundary ) ) { if( atApiBoundary ) apiCallElement = stackTraceElement; break; } candidateUserCodeElement = stackTraceElement; } String userCode = candidateUserCodeElement == null ? "" : candidateUserCodeElement.toString(); String apiCall = ""; if( apiCallElement != null ) { String method = apiCallElement.getMethodName(); if( method.equals( "<init>" ) ) apiCall = String.format( "new %s()", getSimpleClassName( apiCallElement.getClassName() ) ); else apiCall = String.format( "%s()", method ); } return userCode.isEmpty() ? apiCall : apiCall.isEmpty() ? userCode : String.format( "%s @ %s", apiCall, userCode ); } private static Object getSimpleClassName( String className ) { if( className == null || className.isEmpty() ) return ""; String parts[] = className.split( "\\." ); if( parts.length == 0 ) return ""; return parts[ parts.length - 1 ]; } private static boolean atApiBoundary( String stackTraceElement ) { for( Pattern boundary : registeredApiBoundaries.values() ) { if( boundary.matcher( stackTraceElement ).matches() ) return true; } return false; } public static String stringifyStackTrace( Throwable throwable, String lineSeparator, boolean trimLines, int lineLimit ) { if( lineLimit == 0 ) return null; Writer traceWriter = new StringWriter(); PrintWriter printWriter = new PrintWriter( traceWriter ); throwable.printStackTrace( printWriter ); String trace = traceWriter.toString(); if( lineSeparator.equals( System.getProperty( "line.separator" ) ) && !trimLines && lineLimit == -1 ) return trace; lineLimit = lineLimit == -1 ? Integer.MAX_VALUE : lineLimit; StringBuilder buffer = new StringBuilder(); LineNumberReader reader = new LineNumberReader( new StringReader( trace ) ); try { String line = reader.readLine(); while( line != null && reader.getLineNumber() - 1 < lineLimit ) { if( reader.getLineNumber() > 1 ) buffer.append( lineSeparator ); if( trimLines ) line = line.trim(); buffer.append( line ); line = reader.readLine(); } } catch( IOException exception ) { } return buffer.toString(); } }