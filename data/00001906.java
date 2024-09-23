public class RegressionMiscPlatformTest extends PlatformTestCase { public RegressionMiscPlatformTest() { } @Test public void testWriteDot() throws Exception { Tap source = getPlatform().getTextFile( "input" ); Tap sink = getPlatform().getTextFile( "unknown" ); Pipe pipe = new Pipe( "test" ); pipe = new Each( pipe, new Fields( "line" ), new RegexSplitter( Fields.UNKNOWN ) ); pipe = new Each( pipe, new Debug() ); pipe = new Each( pipe, new Fields( 2 ), new Identity( new Fields( "label" ) ) ); pipe = new Each( pipe, new Debug() ); pipe = new Each( pipe, new Fields( "label" ), new RegexFilter( "[A-Z]*" ) ); pipe = new Each( pipe, new Debug() ); pipe = new GroupBy( pipe, Fields.ALL ); pipe = new GroupBy( pipe, Fields.ALL ); Flow flow = getPlatform().getFlowConnector().connect( source, sink, pipe ); String outputPath = getOutputPath( "writedot.dot" ); flow.writeDOT( outputPath ); assertTrue( new File( outputPath ).exists() ); outputPath = getOutputPath( "writestepdot.dot" ); flow.writeStepsDOT( outputPath ); assertTrue( new File( outputPath ).exists() ); } @Test public void testSinkDeclaredFieldsFails() throws IOException { Tap source = getPlatform().getTextFile( new Fields( "line" ), "input" ); Pipe pipe = new Pipe( "test" ); pipe = new Each( pipe, new RegexSplitter( new Fields( "first", "second", "third" ), "\\s" ), Fields.ALL ); Tap sink = getPlatform().getTextFile( new Fields( "line" ), new Fields( "first", "second", "fifth" ), getOutputPath( "output" ), SinkMode.REPLACE ); try { getPlatform().getFlowConnector().connect( source, sink, pipe ); fail( "did not fail on bad sink field names" ); } catch( Exception exception ) { } } @Test public void testTupleEntryNextTwice() throws IOException { Tap tap = getPlatform().getTextFile( inputFileNums10 ); TupleEntryIterator iterator = tap.openForRead( getPlatform().getFlowProcess() ); int count = 0; while( iterator.hasNext() ) { iterator.next(); count++; } assertFalse( iterator.hasNext() ); assertEquals( 10, count ); } @Test public void testTapReplaceOnWrite() throws IOException { String tapPath = getOutputPath( "tapreplace" ); Tap tap = getPlatform().getTextFile( tapPath, SinkMode.KEEP ); TupleEntryCollector collector = tap.openForWrite( getPlatform().getFlowProcess() ); for( int i = 0; i < 100; i++ ) collector.add( new Tuple( "string", "" + i, i ) ); collector.close(); tap = getPlatform().getTextFile( tapPath, SinkMode.REPLACE ); collector = tap.openForWrite( getPlatform().getFlowProcess() ); for( int i = 0; i < 100; i++ ) collector.add( new Tuple( "string", "" + i, i ) ); collector.close(); } }