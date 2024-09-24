public class TextLine extends Scheme<Configuration, RecordReader, OutputCollector, Object[], Object[]> implements FileFormat { public enum Compress { DEFAULT, ENABLE, DISABLE } public static final String DEFAULT_CHARSET = "UTF-8"; private static final long serialVersionUID = 1L; public static final Fields DEFAULT_SOURCE_FIELDS = new Fields( "offset", "line" ).applyTypes( Long.TYPE, String.class ); private static final Pattern zipPattern = Pattern.compile( "\\.[zZ][iI][pP]([ ,]|$)" ); Compress sinkCompression = Compress.DISABLE; String charsetName = DEFAULT_CHARSET; public TextLine() { super( DEFAULT_SOURCE_FIELDS ); } @ConstructorProperties({"numSinkParts"}) public TextLine( int numSinkParts ) { super( DEFAULT_SOURCE_FIELDS, numSinkParts ); } @ConstructorProperties({"sinkCompression"}) public TextLine( Compress sinkCompression ) { super( DEFAULT_SOURCE_FIELDS ); setSinkCompression( sinkCompression ); } @ConstructorProperties({"sourceFields", "sinkFields"}) public TextLine( Fields sourceFields, Fields sinkFields ) { super( sourceFields, sinkFields ); verify( sourceFields ); } @ConstructorProperties({"sourceFields", "sinkFields", "charsetName"}) public TextLine( Fields sourceFields, Fields sinkFields, String charsetName ) { super( sourceFields, sinkFields ); setCharsetName( charsetName ); verify( sourceFields ); } @ConstructorProperties({"sourceFields", "sinkFields", "numSinkParts"}) public TextLine( Fields sourceFields, Fields sinkFields, int numSinkParts ) { super( sourceFields, sinkFields, numSinkParts ); verify( sourceFields ); } @ConstructorProperties({"sourceFields", "sinkFields", "sinkCompression"}) public TextLine( Fields sourceFields, Fields sinkFields, Compress sinkCompression ) { super( sourceFields, sinkFields ); setSinkCompression( sinkCompression ); verify( sourceFields ); } @ConstructorProperties({"sourceFields", "sinkFields", "sinkCompression", "charsetName"}) public TextLine( Fields sourceFields, Fields sinkFields, Compress sinkCompression, String charsetName ) { super( sourceFields, sinkFields ); setSinkCompression( sinkCompression ); setCharsetName( charsetName ); verify( sourceFields ); } @ConstructorProperties({"sourceFields", "sinkFields", "sinkCompression", "numSinkParts"}) public TextLine( Fields sourceFields, Fields sinkFields, Compress sinkCompression, int numSinkParts ) { super( sourceFields, sinkFields, numSinkParts ); setSinkCompression( sinkCompression ); verify( sourceFields ); } @ConstructorProperties({"sourceFields", "sinkFields", "sinkCompression", "numSinkParts", "charsetName"}) public TextLine( Fields sourceFields, Fields sinkFields, Compress sinkCompression, int numSinkParts, String charsetName ) { super( sourceFields, sinkFields, numSinkParts ); setSinkCompression( sinkCompression ); setCharsetName( charsetName ); verify( sourceFields ); } @ConstructorProperties({"sourceFields"}) public TextLine( Fields sourceFields ) { super( sourceFields ); verify( sourceFields ); } @ConstructorProperties({"sourceFields", "charsetName"}) public TextLine( Fields sourceFields, String charsetName ) { super( sourceFields ); setCharsetName( charsetName ); verify( sourceFields ); } @ConstructorProperties({"sourceFields", "numSinkParts"}) public TextLine( Fields sourceFields, int numSinkParts ) { super( sourceFields, numSinkParts ); verify( sourceFields ); } protected void setCharsetName( String charsetName ) { if( charsetName != null ) this.charsetName = charsetName; Charset.forName( this.charsetName ); } @Property(name = "charset", visibility = Visibility.PUBLIC) @PropertyDescription(value = "character set used in this scheme.") public String getCharsetName() { return charsetName; } protected void verify( Fields sourceFields ) { if( sourceFields.size() < 1 || sourceFields.size() > 2 ) throw new IllegalArgumentException( "this scheme requires either one or two source fields, given [" + sourceFields + "]" ); } @Property(name = "sinkCompression", visibility = Visibility.PUBLIC) @PropertyDescription(value = "The compression of the scheme when used in a sink.") public Compress getSinkCompression() { return sinkCompression; } public void setSinkCompression( Compress sinkCompression ) { if( sinkCompression != null ) this.sinkCompression = sinkCompression; } @Override public void sourceConfInit( FlowProcess<? extends Configuration> flowProcess, Tap<Configuration, RecordReader, OutputCollector> tap, Configuration conf ) { JobConf jobConf = asJobConfInstance( conf ); String paths = jobConf.get( "mapred.input.dir", "" ); if( hasZippedFiles( paths ) ) throw new IllegalStateException( "cannot read zip files: " + paths ); conf.setBoolean( "mapred.mapper.new-api", false ); conf.setClass( "mapred.input.format.class", TextInputFormat.class, InputFormat.class ); } private boolean hasZippedFiles( String paths ) { if( paths == null || paths.length() == 0 ) return false; return zipPattern.matcher( paths ).find(); } @Override public void presentSourceFields( FlowProcess<? extends Configuration> flowProcess, Tap tap, Fields fields ) { } @Override public void presentSinkFields( FlowProcess<? extends Configuration> flowProcess, Tap tap, Fields fields ) { } @Override public void sinkConfInit( FlowProcess<? extends Configuration> flowProcess, Tap<Configuration, RecordReader, OutputCollector> tap, Configuration conf ) { if( tap.getFullIdentifier( conf ).endsWith( ".zip" ) ) throw new IllegalStateException( "cannot write zip files: " + HadoopUtil.getOutputPath( conf ) ); conf.setBoolean( "mapred.mapper.new-api", false ); if( getSinkCompression() == Compress.DISABLE ) conf.setBoolean( "mapred.output.compress", false ); else if( getSinkCompression() == Compress.ENABLE ) conf.setBoolean( "mapred.output.compress", true ); conf.setClass( "mapred.output.key.class", Text.class, Object.class ); conf.setClass( "mapred.output.value.class", Text.class, Object.class ); conf.setClass( "mapred.output.format.class", TextOutputFormat.class, OutputFormat.class ); } @Override public void sourcePrepare( FlowProcess<? extends Configuration> flowProcess, SourceCall<Object[], RecordReader> sourceCall ) { if( sourceCall.getContext() == null ) sourceCall.setContext( new Object[ 3 ] ); sourceCall.getContext()[ 0 ] = sourceCall.getInput().createKey(); sourceCall.getContext()[ 1 ] = sourceCall.getInput().createValue(); sourceCall.getContext()[ 2 ] = Charset.forName( charsetName ); } @Override public boolean source( FlowProcess<? extends Configuration> flowProcess, SourceCall<Object[], RecordReader> sourceCall ) throws IOException { if( !sourceReadInput( sourceCall ) ) return false; sourceHandleInput( sourceCall ); return true; } private boolean sourceReadInput( SourceCall<Object[], RecordReader> sourceCall ) throws IOException { Object[] context = sourceCall.getContext(); return sourceCall.getInput().next( context[ 0 ], context[ 1 ] ); } protected void sourceHandleInput( SourceCall<Object[], RecordReader> sourceCall ) throws IOException { TupleEntry result = sourceCall.getIncomingEntry(); int index = 0; Object[] context = sourceCall.getContext(); if( getSourceFields().size() == 2 ) result.setLong( index++, ( (LongWritable) context[ 0 ] ).get() ); result.setString( index, makeEncodedString( context ) ); } protected String makeEncodedString( Object[] context ) { Text text = (Text) context[ 1 ]; return new String( text.getBytes(), 0, text.getLength(), (Charset) context[ 2 ] ); } @Override public void sourceCleanup( FlowProcess<? extends Configuration> flowProcess, SourceCall<Object[], RecordReader> sourceCall ) { sourceCall.setContext( null ); } @Override public void sinkPrepare( FlowProcess<? extends Configuration> flowProcess, SinkCall<Object[], OutputCollector> sinkCall ) throws IOException { sinkCall.setContext( new Object[ 2 ] ); sinkCall.getContext()[ 0 ] = new Text(); sinkCall.getContext()[ 1 ] = Charset.forName( charsetName ); } @Override public void sink( FlowProcess<? extends Configuration> flowProcess, SinkCall<Object[], OutputCollector> sinkCall ) throws IOException { Text text = (Text) sinkCall.getContext()[ 0 ]; Charset charset = (Charset) sinkCall.getContext()[ 1 ]; String line = sinkCall.getOutgoingEntry().getTuple().toString(); text.set( line.getBytes( charset ) ); sinkCall.getOutput().collect( null, text ); } @Override public String getExtension() { return "txt"; } }