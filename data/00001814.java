public class CascadingStatsLocalHadoopErrorPlatformTest extends PlatformTestCase { public CascadingStatsLocalHadoopErrorPlatformTest() { super( false ); } @Before public void setUp() { HadoopFlowStepJob.reportLocalError( null ); } @After public void tearDown() { HadoopFlowStepJob.reportLocalError( null ); } public class FailFunction extends BaseOperation implements Function { public FailFunction( Fields fieldDeclaration ) { super( 1, fieldDeclaration ); } @Override public void operate( FlowProcess flowProcess, FunctionCall functionCall ) { throw new CascadingException( "testing" ); } } @Test public void testLocalErrorReportingInMapper() throws Exception { getPlatform().copyFromLocal( inputFileApache ); Tap source = getPlatform().getTextFile( inputFileApache ); Pipe pipe = new Pipe( "failing mapper" ); pipe = new Each( pipe, new Fields( "line" ), new FailFunction( new Fields( "ip" ) ), new Fields( "ip" ) ); Tap sink = getPlatform().getTextFile( getOutputPath( "mapperfail" ), SinkMode.REPLACE ); Flow flow = getPlatform().getFlowConnector().connect( "mapper fail test", source, sink, pipe ); Cascade cascade = new CascadeConnector( getProperties() ).connect( flow ); assertNull( cascade.getCascadeStats().getThrowable() ); try { cascade.complete(); fail( "An exception should have been thrown" ); } catch( Throwable throwable ) { CascadeStats cascadeStats = cascade.getCascadeStats(); assertEquals( throwable, cascadeStats.getThrowable() ); } } @Test public void testLocalErrorReportingInReducer() throws Exception { getPlatform().copyFromLocal( inputFileApache ); Tap source = getPlatform().getTextFile( inputFileApache ); Pipe pipe = new Pipe( "failing reducer" ); pipe = new Each( pipe, new Fields( "line" ), new RegexParser( new Fields( "ip" ), "^[^ ]*" ), new Fields( "ip" ) ); pipe = new GroupBy( pipe, new Fields( "ip" ) ); pipe = new Every( pipe, new TestFailAggregator( new Fields( "count" ), 1 ) ); Tap sink = getPlatform().getTextFile( getOutputPath( "reducerfail" ), SinkMode.REPLACE ); Flow flow = getPlatform().getFlowConnector().connect( "reducer fail test", source, sink, pipe ); Cascade cascade = new CascadeConnector( getProperties() ).connect( flow ); assertNull( cascade.getCascadeStats().getThrowable() ); try { cascade.complete(); fail( "An exception should have been thrown" ); } catch( Throwable throwable ) { CascadeStats cascadeStats = cascade.getCascadeStats(); assertEquals( throwable, cascadeStats.getThrowable() ); } } }