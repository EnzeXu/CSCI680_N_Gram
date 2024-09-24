public class ParquetTupleSchemeTest { public static final String BUILD = System.getProperty( "test.output.root", "build/test/output" ); final String parquetInputPath = BUILD + "/ParquetTupleIn/names-parquet-in"; final String txtOutputPath = BUILD + "/ParquetTupleOut/names-txt-out"; @Test public void testReadPattern() throws Exception { String sourceFolder = parquetInputPath; testReadWrite( sourceFolder ); String sourceGlobPattern = parquetInputPath + "*"; testReadWrite( multiLevelGlobPattern ); } @Test public void testFieldProjection() throws Exception { createFileForRead(); Path path = new Path( txtOutputPath ); final FileSystem fs = path.getFileSystem( new Configuration() ); if( fs.exists( path ) ) fs.delete( path, true ); Scheme sourceScheme = new ParquetTupleScheme( new Fields( "last_name" ) ); Tap source = new Hfs( sourceScheme, parquetInputPath ); Scheme sinkScheme = new TextLine( new Fields( "last_name" ) ); Tap sink = new Hfs( sinkScheme, txtOutputPath ); Pipe assembly = new Pipe( "namecp" ); assembly = new Each( assembly, new ProjectedTupleFunction() ); Flow flow = new Hadoop3MRFlowConnector().connect( "namecp", source, sink, assembly ); flow.complete(); String result = FileUtils.readFileToString( new File( txtOutputPath + "/part-00000" ) ); assertEquals( "Practice\nHope\nHorse\n", result ); } public void testReadWrite( String inputPath ) throws Exception { createFileForRead(); Path path = new Path( txtOutputPath ); final FileSystem fs = path.getFileSystem( new Configuration() ); if( fs.exists( path ) ) fs.delete( path, true ); Scheme sourceScheme = new ParquetTupleScheme( new Fields( "first_name", "last_name" ) ); Tap source = new Hfs( sourceScheme, inputPath ); Scheme sinkScheme = new TextLine( new Fields( "first", "last" ) ); Tap sink = new Hfs( sinkScheme, txtOutputPath ); Pipe assembly = new Pipe( "namecp" ); assembly = new Each( assembly, new UnpackTupleFunction() ); Flow flow = new Hadoop3MRFlowConnector().connect( "namecp", source, sink, assembly ); flow.complete(); String result = FileUtils.readFileToString( new File( txtOutputPath + "/part-00000" ) ); assertEquals( "Alice\tPractice\nBob\tHope\nCharlie\tHorse\n", result ); } private void createFileForRead() throws Exception { final Path fileToCreate = new Path( parquetInputPath + "/names.parquet" ); final Configuration conf = new Configuration(); final FileSystem fs = fileToCreate.getFileSystem( conf ); if( fs.exists( fileToCreate ) ) fs.delete( fileToCreate, true ); TProtocolFactory protocolFactory = new TCompactProtocol.Factory(); TaskAttemptID taskId = new TaskAttemptID( "local", 0, true, 0, 0 ); ThriftToParquetFileWriter w = new ThriftToParquetFileWriter( fileToCreate, ContextUtil.newTaskAttemptContext( conf, taskId ), protocolFactory, Name.class ); final ByteArrayOutputStream baos = new ByteArrayOutputStream(); final TProtocol protocol = protocolFactory.getProtocol( new TIOStreamTransport( baos ) ); Name n1 = new Name(); n1.setFirst_name( "Alice" ); n1.setLast_name( "Practice" ); Name n2 = new Name(); n2.setFirst_name( "Bob" ); n2.setLast_name( "Hope" ); Name n3 = new Name(); n3.setFirst_name( "Charlie" ); n3.setLast_name( "Horse" ); n1.write( protocol ); w.write( new BytesWritable( baos.toByteArray() ) ); baos.reset(); n2.write( protocol ); w.write( new BytesWritable( baos.toByteArray() ) ); baos.reset(); n3.write( protocol ); w.write( new BytesWritable( baos.toByteArray() ) ); w.close(); } private static class UnpackTupleFunction extends BaseOperation implements Function { @Override public void operate( FlowProcess flowProcess, FunctionCall functionCall ) { TupleEntry arguments = functionCall.getArguments(); Tuple result = new Tuple(); Tuple name = new Tuple(); name.addString( arguments.getString( 0 ) ); name.addString( arguments.getString( 1 ) ); result.add( name ); functionCall.getOutputCollector().add( result ); } } private static class ProjectedTupleFunction extends BaseOperation implements Function { @Override public void operate( FlowProcess flowProcess, FunctionCall functionCall ) { TupleEntry arguments = functionCall.getArguments(); Tuple result = new Tuple(); Tuple name = new Tuple(); name.addString( arguments.getString( 0 ) ); result.add( name ); functionCall.getOutputCollector().add( result ); } } }