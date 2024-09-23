public class CascadingStatsPlatformTest extends PlatformTestCase { enum TestEnum { FIRST, SECOND, THIRD } public CascadingStatsPlatformTest() { super( true ); } @Test public void testStatsCounters() throws Exception { getPlatform().copyFromLocal( inputFileApache ); Tap source = getPlatform().getTextFile( inputFileApache ); Pipe pipe = new Pipe( "first" ); pipe = new Each( pipe, new Fields( "line" ), new RegexParser( new Fields( "ip" ), "^[^ ]*" ), new Fields( "ip" ) ); pipe = new GroupBy( pipe, new Fields( "ip" ) ); pipe = new Each( pipe, new Counter( TestEnum.FIRST ) ); pipe = new GroupBy( pipe, new Fields( "ip" ) ); pipe = new Each( pipe, new Counter( TestEnum.FIRST ) ); pipe = new Each( pipe, new Counter( TestEnum.SECOND ) ); Tap sink1 = getPlatform().getTextFile( getOutputPath( "flowstats1" ), SinkMode.REPLACE ); Tap sink2 = getPlatform().getTextFile( getOutputPath( "flowstats2" ), SinkMode.REPLACE ); Map<Object, Object> properties = getProperties(); properties = FlowRuntimeProps.flowRuntimeProps() .addLogCounter( SliceCounters.Tuples_Read ) .buildProperties( properties ); FlowConnector flowConnector = getPlatform().getFlowConnector( properties ); Flow flow1 = flowConnector.connect( "stats1 test", source, sink1, pipe ); Flow flow2 = flowConnector.connect( "stats2 test", source, sink2, pipe ); Cascade cascade = new CascadeConnector( getProperties() ).connect( flow1, flow2 ); cascade.complete(); CascadeStats cascadeStats = cascade.getCascadeStats(); assertNotNull( cascadeStats.getID() ); assertEquals( 2, cascadeStats.getCountersFor( TestEnum.class.getName() ).size() ); assertEquals( 2, cascadeStats.getCountersFor( TestEnum.class ).size() ); assertEquals( 40, cascadeStats.getCounterValue( TestEnum.FIRST ) ); assertEquals( 20, cascadeStats.getCounterValue( TestEnum.SECOND ) ); assertEquals( 0, cascadeStats.getCounterValue( TestEnum.THIRD ) ); assertEquals( 0, cascadeStats.getCounterValue( "FOO", "BAR" ) ); FlowStats flowStats1 = flow1.getFlowStats(); assertNotNull( flowStats1.getID() ); assertEquals( 20, flowStats1.getCounterValue( TestEnum.FIRST ) ); assertEquals( 10, flowStats1.getCounterValue( TestEnum.SECOND ) ); assertEquals( 0, flowStats1.getCounterValue( TestEnum.THIRD ) ); assertEquals( 0, flowStats1.getCounterValue( "FOO", "BAR" ) ); FlowStats flowStats2 = flow2.getFlowStats(); assertNotNull( flowStats2.getID() ); assertEquals( 20, flowStats2.getCounterValue( TestEnum.FIRST ) ); assertEquals( 10, flowStats2.getCounterValue( TestEnum.SECOND ) ); cascadeStats.captureDetail(); } @Test public void testStatsOnJoin() throws Exception { getPlatform().copyFromLocal( inputFileLower ); getPlatform().copyFromLocal( inputFileUpper ); Tap sourceLower = getPlatform().getTextFile( new Fields( "offset", "line" ), inputFileLower ); Tap sourceUpper = getPlatform().getTextFile( new Fields( "offset", "line" ), inputFileUpper ); Map sources = new HashMap(); sources.put( "lower", sourceLower ); sources.put( "upper", sourceUpper ); Tap sink = getPlatform().getTextFile( new Fields( "line" ), getOutputPath( "join" ), SinkMode.REPLACE ); Function splitter = new RegexSplitter( new Fields( "num", "char" ), " " ); Pipe pipeLower = new Each( new Pipe( "lower" ), new Fields( "line" ), splitter ); pipeLower = new Each( pipeLower, new Counter( TestEnum.FIRST ) ); Pipe pipeUpper = new Each( new Pipe( "upper" ), new Fields( "line" ), splitter ); pipeUpper = new Each( pipeUpper, new Counter( TestEnum.SECOND ) ); Pipe splice = new HashJoin( pipeLower, new Fields( "num" ), pipeUpper, new Fields( "num" ), Fields.size( 4 ) ); Map<Object, Object> properties = getProperties(); Flow flow = getPlatform().getFlowConnector( properties ).connect( sources, sink, splice ); flow.complete(); validateLength( flow, 5 ); FlowStats flowStats = flow.getFlowStats(); assertNotNull( flowStats.getID() ); long firstCounter = flowStats.getCounterValue( TestEnum.FIRST ); long secondCounter = flowStats.getCounterValue( TestEnum.SECOND ); assertEquals( 5, firstCounter ); assertNotSame( 0, secondCounter ); assertEquals( firstCounter + secondCounter, flowStats.getCounterValue( SliceCounters.Tuples_Read ) ); } }