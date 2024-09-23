public class LocalFailScheme extends cascading.scheme.local.TextLine { boolean sourceFired = false; boolean sinkFired = false; public LocalFailScheme() { } public LocalFailScheme( Fields sourceFields ) { super( sourceFields ); } @Override public boolean source( FlowProcess<? extends Properties> flowProcess, SourceCall<LineNumberReader, InputStream> sourceCall ) throws IOException { if( !sourceFired ) { sourceFired = true; throw new TapException( "fail", new Tuple( "bad data" ) ); } return super.source( flowProcess, sourceCall ); } @Override public void sink( FlowProcess<? extends Properties> flowProcess, SinkCall<PrintWriter, OutputStream> sinkCall ) throws IOException { if( !sinkFired ) { sinkFired = true; throw new TapException( "fail", new Tuple( "bad data" ) ); } super.sink( flowProcess, sinkCall ); } }