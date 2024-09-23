public class TempHfs extends Hfs { final String name; private Class<? extends Scheme> schemeClass; private static class NullScheme extends Scheme<Configuration, RecordReader, OutputCollector, Object, Object> { @Override public void sourceConfInit( FlowProcess<? extends Configuration> flowProcess, Tap<Configuration, RecordReader, OutputCollector> tap, Configuration conf ) { } @Override public void sinkConfInit( FlowProcess<? extends Configuration> flowProcess, Tap<Configuration, RecordReader, OutputCollector> tap, Configuration conf ) { conf.setClass( "mapred.output.key.class", Tuple.class, Object.class ); conf.setClass( "mapred.output.value.class", Tuple.class, Object.class ); conf.setClass( "mapred.output.format.class", NullOutputFormat.class, OutputFormat.class ); } @Override public boolean source( FlowProcess<? extends Configuration> flowProcess, SourceCall<Object, RecordReader> sourceCall ) throws IOException { return false; } @Override public void sink( FlowProcess<? extends Configuration> flowProcess, SinkCall<Object, OutputCollector> sinkCall ) throws IOException { } } public TempHfs( Configuration conf, String name, boolean isNull ) { super( isNull ? new NullScheme() : new SequenceFile() { } ); this.name = name; this.stringPath = initTemporaryPath( conf, true ); } public TempHfs( Configuration conf, String name, Class<? extends Scheme> schemeClass ) { this( conf, name, schemeClass, true ); } public TempHfs( Configuration conf, String name, Class<? extends Scheme> schemeClass, boolean unique ) { this.name = name; if( schemeClass == null ) this.schemeClass = SequenceFile.class; else this.schemeClass = schemeClass; this.stringPath = initTemporaryPath( conf, unique ); } public Class<? extends Scheme> getSchemeClass() { return schemeClass; } private String initTemporaryPath( Configuration conf, boolean unique ) { String child = unique ? makeTemporaryPathDirString( name ) : name; return new Path( getTempPath( conf ), child ).toString(); } @Override public Scope outgoingScopeFor( Set<Scope> incomingScopes ) { Fields fields = incomingScopes.iterator().next().getIncomingTapFields(); setSchemeUsing( fields ); return new Scope( fields ); } private void setSchemeUsing( Fields fields ) { try { setScheme( schemeClass.getConstructor( Fields.class ).newInstance( fields ) ); } catch( Exception exception ) { throw new CascadingException( "unable to create specified scheme: " + schemeClass.getName(), exception ); } } @Override public boolean isTemporary() { return true; } @Override public String toString() { return getClass().getSimpleName() + "[\"" + getScheme() + "\"]" + "[" + name + "]"; } }