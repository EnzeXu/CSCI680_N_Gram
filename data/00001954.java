public class TextDelimited extends TextLine { public static final String DEFAULT_CHARSET = "UTF-8"; protected final DelimitedParser delimitedParser; private boolean skipHeader; private final boolean writeHeader; public TextDelimited() { this( Fields.ALL, null, "\t", null, null ); } @ConstructorProperties({"hasHeader", "delimiter"}) public TextDelimited( boolean hasHeader, String delimiter ) { this( Fields.ALL, null, hasHeader, delimiter, null, (Class[]) null ); } @ConstructorProperties({"hasHeader", "delimiter", "quote"}) public TextDelimited( boolean hasHeader, String delimiter, String quote ) { this( Fields.ALL, null, hasHeader, delimiter, quote, (Class[]) null ); } @ConstructorProperties({"hasHeader", "delimitedParser"}) public TextDelimited( boolean hasHeader, DelimitedParser delimitedParser ) { this( Fields.ALL, null, hasHeader, hasHeader, delimitedParser ); } @ConstructorProperties({"delimitedParser"}) public TextDelimited( DelimitedParser delimitedParser ) { this( Fields.ALL, null, true, true, delimitedParser ); } @ConstructorProperties({"sinkCompression", "hasHeader", "delimitedParser"}) public TextDelimited( Compress sinkCompression, boolean hasHeader, DelimitedParser delimitedParser ) { this( Fields.ALL, sinkCompression, hasHeader, hasHeader, delimitedParser ); } @ConstructorProperties({"sinkCompression", "delimitedParser"}) public TextDelimited( Compress sinkCompression, DelimitedParser delimitedParser ) { this( Fields.ALL, sinkCompression, true, true, delimitedParser ); } @ConstructorProperties({"sinkCompression", "hasHeader", "delimiter", "quote"}) public TextDelimited( Compress sinkCompression, boolean hasHeader, String delimiter, String quote ) { this( Fields.ALL, sinkCompression, hasHeader, delimiter, quote, (Class[]) null ); } @ConstructorProperties({"fields"}) public TextDelimited( Fields fields ) { this( fields, null, "\t", null, null ); } @ConstructorProperties({"fields", "delimiter"}) public TextDelimited( Fields fields, String delimiter ) { this( fields, null, delimiter, null, null ); } @ConstructorProperties({"fields", "hasHeader", "delimiter"}) public TextDelimited( Fields fields, boolean hasHeader, String delimiter ) { this( fields, null, hasHeader, hasHeader, delimiter, null, null ); } @ConstructorProperties({"fields", "skipHeader", "writeHeader", "delimiter"}) public TextDelimited( Fields fields, boolean skipHeader, boolean writeHeader, String delimiter ) { this( fields, null, skipHeader, writeHeader, delimiter, null, null ); } @ConstructorProperties({"fields", "delimiter", "types"}) public TextDelimited( Fields fields, String delimiter, Class[] types ) { this( fields, null, delimiter, null, types ); } @ConstructorProperties({"fields", "hasHeader", "delimiter", "types"}) public TextDelimited( Fields fields, boolean hasHeader, String delimiter, Class[] types ) { this( fields, null, hasHeader, hasHeader, delimiter, null, types ); } @ConstructorProperties({"fields", "skipHeader", "writeHeader", "delimiter", "types"}) public TextDelimited( Fields fields, boolean skipHeader, boolean writeHeader, String delimiter, Class[] types ) { this( fields, null, skipHeader, writeHeader, delimiter, null, types ); } @ConstructorProperties({"fields", "delimiter", "quote", "types"}) public TextDelimited( Fields fields, String delimiter, String quote, Class[] types ) { this( fields, null, delimiter, quote, types ); } @ConstructorProperties({"fields", "hasHeader", "delimiter", "quote", "types"}) public TextDelimited( Fields fields, boolean hasHeader, String delimiter, String quote, Class[] types ) { this( fields, null, hasHeader, hasHeader, delimiter, quote, types ); } @ConstructorProperties({"fields", "skipHeader", "writeHeader", "delimiter", "quote", "types"}) public TextDelimited( Fields fields, boolean skipHeader, boolean writeHeader, String delimiter, String quote, Class[] types ) { this( fields, null, skipHeader, writeHeader, delimiter, quote, types ); } @ConstructorProperties({"fields", "delimiter", "quote", "types", "safe"}) public TextDelimited( Fields fields, String delimiter, String quote, Class[] types, boolean safe ) { this( fields, null, delimiter, quote, types, safe ); } @ConstructorProperties({"fields", "hasHeader", "delimiter", "quote", "types", "safe"}) public TextDelimited( Fields fields, boolean hasHeader, String delimiter, String quote, Class[] types, boolean safe ) { this( fields, null, hasHeader, hasHeader, delimiter, quote, types, safe ); } @ConstructorProperties({"fields", "hasHeader", "delimiter", "quote", "types", "safe", "charsetName"}) public TextDelimited( Fields fields, boolean hasHeader, String delimiter, String quote, Class[] types, boolean safe, String charsetName ) { this( fields, null, hasHeader, hasHeader, delimiter, true, quote, types, safe, charsetName ); } @ConstructorProperties({"fields", "skipHeader", "writeHeader", "delimiter", "quote", "types", "safe"}) public TextDelimited( Fields fields, boolean skipHeader, boolean writeHeader, String delimiter, String quote, Class[] types, boolean safe ) { this( fields, null, skipHeader, writeHeader, delimiter, quote, types, safe ); } @ConstructorProperties({"fields", "sinkCompression", "delimiter"}) public TextDelimited( Fields fields, Compress sinkCompression, String delimiter ) { this( fields, sinkCompression, delimiter, null, null ); } @ConstructorProperties({"fields", "sinkCompression", "hasHeader", "delimiter"}) public TextDelimited( Fields fields, Compress sinkCompression, boolean hasHeader, String delimiter ) { this( fields, sinkCompression, hasHeader, hasHeader, delimiter, null, null ); } @ConstructorProperties({"fields", "sinkCompression", "skipHeader", "writeHeader", "delimiter"}) public TextDelimited( Fields fields, Compress sinkCompression, boolean skipHeader, boolean writeHeader, String delimiter ) { this( fields, sinkCompression, skipHeader, writeHeader, delimiter, null, null ); } @ConstructorProperties({"fields", "sinkCompression", "delimiter", "types"}) public TextDelimited( Fields fields, Compress sinkCompression, String delimiter, Class[] types ) { this( fields, sinkCompression, delimiter, null, types ); } @ConstructorProperties({"fields", "sinkCompression", "hasHeader", "delimiter", "types"}) public TextDelimited( Fields fields, Compress sinkCompression, boolean hasHeader, String delimiter, Class[] types ) { this( fields, sinkCompression, hasHeader, hasHeader, delimiter, null, types ); } @ConstructorProperties({"fields", "sinkCompression", "skipHeader", "writeHeader", "delimiter", "types"}) public TextDelimited( Fields fields, Compress sinkCompression, boolean skipHeader, boolean writeHeader, String delimiter, Class[] types ) { this( fields, sinkCompression, skipHeader, writeHeader, delimiter, null, types ); } @ConstructorProperties({"fields", "sinkCompression", "delimiter", "types", "safe"}) public TextDelimited( Fields fields, Compress sinkCompression, String delimiter, Class[] types, boolean safe ) { this( fields, sinkCompression, delimiter, null, types, safe ); } @ConstructorProperties({"fields", "sinkCompression", "hasHeader", "delimiter", "types", "safe"}) public TextDelimited( Fields fields, Compress sinkCompression, boolean hasHeader, String delimiter, Class[] types, boolean safe ) { this( fields, sinkCompression, hasHeader, hasHeader, delimiter, null, types, safe ); } @ConstructorProperties({"fields", "sinkCompression", "hasHeader", "delimiter", "types", "safe", "charsetName"}) public TextDelimited( Fields fields, Compress sinkCompression, boolean hasHeader, String delimiter, Class[] types, boolean safe, String charsetName ) { this( fields, sinkCompression, hasHeader, hasHeader, delimiter, true, null, types, safe, charsetName ); } @ConstructorProperties({"fields", "sinkCompression", "skipHeader", "writeHeader", "delimiter", "types", "safe"}) public TextDelimited( Fields fields, Compress sinkCompression, boolean skipHeader, boolean writeHeader, String delimiter, Class[] types, boolean safe ) { this( fields, sinkCompression, skipHeader, writeHeader, delimiter, null, types, safe ); } @ConstructorProperties({"fields", "delimiter", "quote"}) public TextDelimited( Fields fields, String delimiter, String quote ) { this( fields, null, delimiter, quote ); } @ConstructorProperties({"fields", "hasHeader", "delimiter", "quote"}) public TextDelimited( Fields fields, boolean hasHeader, String delimiter, String quote ) { this( fields, null, hasHeader, hasHeader, delimiter, quote ); } @ConstructorProperties({"fields", "skipHeader", "writeHeader", "delimiter", "quote"}) public TextDelimited( Fields fields, boolean skipHeader, boolean writeHeader, String delimiter, String quote ) { this( fields, null, skipHeader, writeHeader, delimiter, quote ); } @ConstructorProperties({"fields", "sinkCompression", "delimiter", "quote"}) public TextDelimited( Fields fields, Compress sinkCompression, String delimiter, String quote ) { this( fields, sinkCompression, false, false, delimiter, true, quote, null, true ); } @ConstructorProperties({"fields", "sinkCompression", "hasHeader", "delimiter", "quote"}) public TextDelimited( Fields fields, Compress sinkCompression, boolean hasHeader, String delimiter, String quote ) { this( fields, sinkCompression, hasHeader, hasHeader, delimiter, true, quote, null, true ); } @ConstructorProperties({"fields", "sinkCompression", "hasHeader", "delimiter", "quote", "charsetName"}) public TextDelimited( Fields fields, Compress sinkCompression, boolean hasHeader, String delimiter, String quote, String charsetName ) { this( fields, sinkCompression, hasHeader, hasHeader, delimiter, true, quote, null, true, charsetName ); } @ConstructorProperties({"fields", "sinkCompression", "skipHeader", "writeHeader", "delimiter", "quote"}) public TextDelimited( Fields fields, Compress sinkCompression, boolean skipHeader, boolean writeHeader, String delimiter, String quote ) { this( fields, sinkCompression, skipHeader, writeHeader, delimiter, true, quote, null, true ); } @ConstructorProperties({"fields", "sinkCompression", "delimiter", "quote", "types"}) public TextDelimited( Fields fields, Compress sinkCompression, String delimiter, String quote, Class[] types ) { this( fields, sinkCompression, false, false, delimiter, true, quote, types, true ); } @ConstructorProperties({"fields", "sinkCompression", "hasHeader", "delimiter", "quote", "types"}) public TextDelimited( Fields fields, Compress sinkCompression, boolean hasHeader, String delimiter, String quote, Class[] types ) { this( fields, sinkCompression, hasHeader, hasHeader, delimiter, true, quote, types, true ); } @ConstructorProperties({"fields", "sinkCompression", "skipHeader", "writeHeader", "delimiter", "quote", "types"}) public TextDelimited( Fields fields, Compress sinkCompression, boolean skipHeader, boolean writeHeader, String delimiter, String quote, Class[] types ) { this( fields, sinkCompression, skipHeader, writeHeader, delimiter, true, quote, types, true ); } @ConstructorProperties({"fields", "sinkCompression", "delimiter", "quote", "types", "safe"}) public TextDelimited( Fields fields, Compress sinkCompression, String delimiter, String quote, Class[] types, boolean safe ) { this( fields, sinkCompression, false, false, delimiter, true, quote, types, safe ); } @ConstructorProperties({"fields", "sinkCompression", "hasHeader", "delimiter", "quote", "types", "safe"}) public TextDelimited( Fields fields, Compress sinkCompression, boolean hasHeader, String delimiter, String quote, Class[] types, boolean safe ) { this( fields, sinkCompression, hasHeader, hasHeader, delimiter, true, quote, types, safe ); } @ConstructorProperties({"fields", "sinkCompression", "skipHeader", "writeHeader", "delimiter", "quote", "types", "safe"}) public TextDelimited( Fields fields, Compress sinkCompression, boolean skipHeader, boolean writeHeader, String delimiter, String quote, Class[] types, boolean safe ) { this( fields, sinkCompression, skipHeader, writeHeader, delimiter, true, quote, types, safe ); } @ConstructorProperties({"fields", "sinkCompression", "skipHeader", "writeHeader", "delimiter", "strict", "quote", "types", "safe"}) public TextDelimited( Fields fields, Compress sinkCompression, boolean skipHeader, boolean writeHeader, String delimiter, boolean strict, String quote, Class[] types, boolean safe ) { this( fields, sinkCompression, skipHeader, writeHeader, delimiter, strict, quote, types, safe, DEFAULT_CHARSET ); } @ConstructorProperties({"fields", "sinkCompression", "skipHeader", "writeHeader", "delimiter", "strict", "quote", "types", "safe", "charsetName"}) public TextDelimited( Fields fields, Compress sinkCompression, boolean skipHeader, boolean writeHeader, String delimiter, boolean strict, String quote, Class[] types, boolean safe, String charsetName ) { this( fields, sinkCompression, skipHeader, writeHeader, charsetName, new DelimitedParser( delimiter, quote, types, strict, safe ) ); } @ConstructorProperties({"fields", "skipHeader", "writeHeader", "delimitedParser"}) public TextDelimited( Fields fields, boolean skipHeader, boolean writeHeader, DelimitedParser delimitedParser ) { this( fields, null, skipHeader, writeHeader, null, delimitedParser ); } @ConstructorProperties({"fields", "hasHeader", "delimitedParser"}) public TextDelimited( Fields fields, boolean hasHeader, DelimitedParser delimitedParser ) { this( fields, null, hasHeader, hasHeader, null, delimitedParser ); } @ConstructorProperties({"fields", "sinkCompression", "skipHeader", "writeHeader", "delimitedParser"}) public TextDelimited( Fields fields, Compress sinkCompression, boolean skipHeader, boolean writeHeader, DelimitedParser delimitedParser ) { this( fields, sinkCompression, skipHeader, writeHeader, null, delimitedParser ); } @ConstructorProperties({"fields", "sinkCompression", "skipHeader", "writeHeader", "charsetName", "delimitedParser"}) public TextDelimited( Fields fields, Compress sinkCompression, boolean skipHeader, boolean writeHeader, String charsetName, DelimitedParser delimitedParser ) { super( sinkCompression ); this.delimitedParser = delimitedParser; setSinkFields( fields ); setSourceFields( fields ); this.skipHeader = skipHeader; this.writeHeader = writeHeader; setCharsetName( charsetName ); } @Property(name = "delimiter", visibility = Visibility.PUBLIC) @PropertyDescription("The delimiter used to separate fields.") public String getDelimiter() { return delimitedParser.getDelimiter(); } @Property(name = "quote", visibility = Visibility.PUBLIC) @PropertyDescription("The string used for quoting.") public String getQuote() { return delimitedParser.getQuote(); } @Override public boolean isSymmetrical() { return super.isSymmetrical() && skipHeader == writeHeader; } @Override public void setSinkFields( Fields sinkFields ) { super.setSourceFields( sinkFields ); super.setSinkFields( sinkFields ); if( delimitedParser != null ) delimitedParser.reset( getSourceFields(), getSinkFields() ); } @Override public void setSourceFields( Fields sourceFields ) { super.setSourceFields( sourceFields ); super.setSinkFields( sourceFields ); if( delimitedParser != null ) delimitedParser.reset( getSourceFields(), getSinkFields() ); } @Override public Fields retrieveSourceFields( FlowProcess<? extends Configuration> flowProcess, Tap tap ) { if( !skipHeader || !getSourceFields().isUnknown() ) return getSourceFields(); if( tap instanceof CompositeTap ) tap = (Tap) ( (CompositeTap) tap ).getChildTaps().next(); if( tap instanceof TapWith ) tap = ( (TapWith) tap ).withScheme( new TextLine( new Fields( "line" ), charsetName ) ).asTap(); else tap = new Hfs( new TextLine( new Fields( "line" ), charsetName ), tap.getFullIdentifier( flowProcess ) ); setSourceFields( delimitedParser.parseFirstLine( flowProcess, tap ) ); return getSourceFields(); } @Override public void presentSourceFields( FlowProcess<? extends Configuration> flowProcess, Tap tap, Fields fields ) { presentSourceFieldsInternal( fields ); } @Override public void presentSinkFields( FlowProcess<? extends Configuration> flowProcess, Tap tap, Fields fields ) { presentSinkFieldsInternal( fields ); } @Override public void sourcePrepare( FlowProcess<? extends Configuration> flowProcess, SourceCall<Object[], RecordReader> sourceCall ) { super.sourcePrepare( flowProcess, sourceCall ); sourceCall.getIncomingEntry().setTuple( TupleViews.createObjectArray() ); } @Override public boolean source( FlowProcess<? extends Configuration> flowProcess, SourceCall<Object[], RecordReader> sourceCall ) throws IOException { Object[] context = sourceCall.getContext(); if( !sourceCall.getInput().next( context[ 0 ], context[ 1 ] ) ) return false; if( skipHeader && ( (LongWritable) context[ 0 ] ).get() == 0 ) { if( !sourceCall.getInput().next( context[ 0 ], context[ 1 ] ) ) return false; } Object[] split = delimitedParser.parseLine( makeEncodedString( context ) ); Tuple tuple = sourceCall.getIncomingEntry().getTuple(); TupleViews.reset( tuple, split ); return true; } @Override public void sinkPrepare( FlowProcess<? extends Configuration> flowProcess, SinkCall<Object[], OutputCollector> sinkCall ) throws IOException { sinkCall.setContext( new Object[ 3 ] ); sinkCall.getContext()[ 0 ] = new Text(); sinkCall.getContext()[ 1 ] = new StringBuilder( 4 * 1024 ); sinkCall.getContext()[ 2 ] = Charset.forName( charsetName ); if( writeHeader ) writeHeader( sinkCall ); } protected void writeHeader( SinkCall<Object[], OutputCollector> sinkCall ) throws IOException { Fields fields = sinkCall.getOutgoingEntry().getFields(); Text text = (Text) sinkCall.getContext()[ 0 ]; StringBuilder line = (StringBuilder) sinkCall.getContext()[ 1 ]; Charset charset = (Charset) sinkCall.getContext()[ 2 ]; line = (StringBuilder) delimitedParser.joinFirstLine( fields, line ); text.set( line.toString().getBytes( charset ) ); sinkCall.getOutput().collect( null, text ); line.setLength( 0 ); } @Override public void sink( FlowProcess<? extends Configuration> flowProcess, SinkCall<Object[], OutputCollector> sinkCall ) throws IOException { TupleEntry tupleEntry = sinkCall.getOutgoingEntry(); Text text = (Text) sinkCall.getContext()[ 0 ]; StringBuilder line = (StringBuilder) sinkCall.getContext()[ 1 ]; Charset charset = (Charset) sinkCall.getContext()[ 2 ]; Iterable<String> strings = tupleEntry.asIterableOf( String.class ); line = (StringBuilder) delimitedParser.joinLine( strings, line ); text.set( line.toString().getBytes( charset ) ); sinkCall.getOutput().collect( null, text ); line.setLength( 0 ); } @Override public String getExtension() { switch( getDelimiter() ) { case "\t": return "tsv"; case ",": return "csv"; } return "txt"; } }