public class FlowSkipIfSinkExists implements FlowSkipStrategy { public boolean skipFlow ( Flow flow ) throws IOException { long sinkModified = flow . getSinkModified ( ) ; if ( sinkModified < = 0 ) return false ; return true ; } }