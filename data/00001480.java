public class TrapProps extends Props { public static final String RECORD_ELEMENT_TRACE = "cascading.trap.elementtrace.record"; public static final String RECORD_THROWABLE_MESSAGE = "cascading.trap.throwable.message.record"; public static final String RECORD_THROWABLE_STACK_TRACE = "cascading.trap.throwable.stacktrace.record"; public static final String LOG_THROWABLE_STACK_TRACE = "cascading.trap.throwable.stacktrace.log"; public static final String STACK_TRACE_LINE_TRIM = "cascading.trap.throwable.stacktrace.line.trim"; public static final String STACK_TRACE_LINE_DELIMITER = "cascading.trap.throwable.stacktrace.line.delimiter"; protected boolean recordElementTrace = false; protected boolean recordThrowableMessage = false; protected boolean recordThrowableStackTrace = false; protected boolean logThrowableStackTrace = true; protected boolean stackTraceTrimLine = true; protected String stackTraceLineDelimiter = null; public static TrapProps trapProps() { return new TrapProps(); } public TrapProps() { } public TrapProps recordAllDiagnostics() { recordElementTrace = true; recordThrowableMessage = true; recordThrowableStackTrace = true; return this; } public boolean isRecordElementTrace() { return recordElementTrace; } public TrapProps setRecordElementTrace( boolean recordElementTrace ) { this.recordElementTrace = recordElementTrace; return this; } public boolean isRecordThrowableMessage() { return recordThrowableMessage; } public TrapProps setRecordThrowableMessage( boolean recordThrowableMessage ) { this.recordThrowableMessage = recordThrowableMessage; return this; } public boolean isRecordThrowableStackTrace() { return recordThrowableStackTrace; } public TrapProps setRecordThrowableStackTrace( boolean recordThrowableStackTrace ) { this.recordThrowableStackTrace = recordThrowableStackTrace; return this; } public boolean isLogThrowableStackTrace() { return logThrowableStackTrace; } public TrapProps setLogThrowableStackTrace( boolean logThrowableStackTrace ) { this.logThrowableStackTrace = logThrowableStackTrace; return this; } public boolean isStackTraceTrimLine() { return stackTraceTrimLine; } public TrapProps setStackTraceTrimLine( boolean stackTraceTrimLine ) { this.stackTraceTrimLine = stackTraceTrimLine; return this; } public String getStackTraceLineDelimiter() { return stackTraceLineDelimiter; } public TrapProps setStackTraceLineDelimiter( String stackTraceLineDelimiter ) { this.stackTraceLineDelimiter = stackTraceLineDelimiter; return this; } @Override protected void addPropertiesTo( Properties properties ) { properties.setProperty( RECORD_ELEMENT_TRACE, Boolean.toString( recordElementTrace ) ); properties.setProperty( RECORD_THROWABLE_MESSAGE, Boolean.toString( recordThrowableMessage ) ); properties.setProperty( RECORD_THROWABLE_STACK_TRACE, Boolean.toString( recordThrowableStackTrace ) ); properties.setProperty( LOG_THROWABLE_STACK_TRACE, Boolean.toString( logThrowableStackTrace ) ); properties.setProperty( STACK_TRACE_LINE_TRIM, Boolean.toString( stackTraceTrimLine ) ); if( stackTraceLineDelimiter != null ) properties.setProperty( STACK_TRACE_LINE_DELIMITER, stackTraceLineDelimiter ); } }