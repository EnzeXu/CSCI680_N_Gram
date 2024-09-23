public class TrapPlatformTest extends PlatformTestCase { public TrapPlatformTest() { super( true, 4, 4 ); } @Test public void testTrapNone() throws Exception { getPlatform().copyFromLocal( inputFileApache ); Tap source = getPlatform().getTextFile( inputFileApache ); Pipe pipe = new Pipe( "map" ); pipe = new Each( pipe, new Fields( "line" ), new RegexParser( new Fields( "ip" ), "^[^ ]*" ), new Fields( "ip" ) ); pipe = new GroupBy( "reduce", pipe, new Fields( "ip" ) ); pipe = new Every( pipe, new Count(), new Fields( "ip", "count" ) ); Tap sink = getPlatform().getTextFile( getOutputPath( "none/tap" ), SinkMode.REPLACE ); Tap trap = getPlatform().getTextFile( getOutputPath( "none/trap" ), SinkMode.REPLACE ); Flow flow = getPlatform().getFlowConnector().connect( "trap test", source, sink, trap, pipe ); flow.complete(); validateLength( flow, 8, null ); try { flow.openTrap(); fail(); } catch( IOException exception ) { } } @Test public void testTrapEachAll() throws Exception { getPlatform().copyFromLocal( inputFileApache ); Tap source = getPlatform().getTextFile( inputFileApache ); Pipe pipe = new Pipe( "map" ); pipe = new Each( pipe, new Fields( "line" ), new RegexParser( new Fields( "ip" ), "^[^ ]*" ), new Fields( "ip" ) ); pipe = new Each( pipe, new Fields( "ip" ), new TestFunction( new Fields( "test" ), null ), Fields.ALL ); pipe = new GroupBy( "reduce", pipe, new Fields( "ip" ) ); pipe = new Every( pipe, new Count(), new Fields( "ip", "count" ) ); Tap sink = getPlatform().getTextFile( getOutputPath( "all/tap" ), SinkMode.REPLACE ); Tap trap = getPlatform().getTextFile( getOutputPath( "all/trap" ), SinkMode.REPLACE ); Flow flow = getPlatform().getFlowConnector().connect( "trap test", source, sink, trap, pipe ); flow.complete(); validateLength( flow, 0, null ); validateLength( flow.openTrap(), 10 ); } @Test public void testTrapEachAllSequence() throws Exception { getPlatform().copyFromLocal( inputFileApache ); Tap source = getPlatform().getTextFile( inputFileApache ); Pipe pipe = new Pipe( "map" ); pipe = new Each( pipe, new Fields( "line" ), new RegexParser( new Fields( "ip" ), "^[^ ]*" ), new Fields( "ip" ) ); pipe = new Each( pipe, new Fields( "ip" ), new TestFunction( new Fields( "test" ), null ), Fields.ALL ); pipe = new GroupBy( "reduce", pipe, new Fields( "ip" ) ); pipe = new Every( pipe, new Count(), new Fields( "ip", "count" ) ); Tap sink = getPlatform().getTabDelimitedFile( Fields.ALL, getOutputPath( "allseq/tap" ), SinkMode.REPLACE ); Tap trap = getPlatform().getTabDelimitedFile( Fields.ALL, getOutputPath( "allseq/trap" ), SinkMode.REPLACE ); Flow flow = getPlatform().getFlowConnector().connect( "trap test", source, sink, trap, pipe ); flow.complete(); validateLength( flow, 0, null ); validateLength( flow.openTrap(), 10 ); } @Test public void testTrapEveryAllAtStart() throws Exception { runTrapEveryAll( 0, "everystart", 8 ); } @Test public void testTrapEveryAllAtAggregate() throws Exception { runTrapEveryAll( 1, "everyaggregate", 10 ); } @Test public void testTrapEveryAllAtComplete() throws Exception { runTrapEveryAll( 2, "everycomplete", 8 ); } private void runTrapEveryAll( int failAt, String path, int failSize ) throws IOException { getPlatform().copyFromLocal( inputFileApache ); Tap source = getPlatform().getTextFile( inputFileApache ); Pipe pipe = new Pipe( "map" ); pipe = new Each( pipe, new Fields( "line" ), new RegexParser( new Fields( "ip" ), "^[^ ]*" ), new Fields( "ip" ) ); pipe = new GroupBy( "reduce", pipe, new Fields( "ip" ) ); pipe = new Every( pipe, new Count(), new Fields( "ip", "count" ) ); pipe = new Every( pipe, new TestFailAggregator( new Fields( "fail" ), failAt ), new Fields( "ip", "count" ) ); Tap sink = getPlatform().getTextFile( getOutputPath( path + "/tap" ), SinkMode.REPLACE ); Tap trap = getPlatform().getTextFile( getOutputPath( path + "/trap" ), SinkMode.REPLACE ); Map<String, Tap> traps = Cascades.tapsMap( "reduce", trap ); Flow flow = getPlatform().getFlowConnector().connect( "trap test", source, sink, traps, pipe ); flow.complete(); validateLength( flow, 0, null ); validateLength( flow.openTrap(), failSize ); } @Test public void testTrapEachAllChained() throws Exception { getPlatform().copyFromLocal( inputFileApache ); Tap source = getPlatform().getTextFile( inputFileApache ); Pipe pipe = new Pipe( "map" ); pipe = new Each( pipe, new Fields( "line" ), new RegexParser( new Fields( "ip" ), "^[^ ]*" ), new Fields( "ip" ) ); pipe = new Each( pipe, new TestFunction( new Fields( "test" ), new Tuple( 1 ), 1 ), Fields.ALL ); pipe = new Each( pipe, new TestFunction( new Fields( "test2" ), new Tuple( 2 ), 2 ), Fields.ALL ); pipe = new Each( pipe, new TestFunction( new Fields( "test3" ), new Tuple( 3 ), 3 ), Fields.ALL ); pipe = new Each( pipe, new TestFunction( new Fields( "test4" ), new Tuple( 4 ), 4 ), Fields.ALL ); Tap sink = getPlatform().getTextFile( getOutputPath( "allchain/tap-nondeterministic" ), SinkMode.REPLACE ); Tap trap = getPlatform().getTextFile( getOutputPath( "allchain/trap-nondeterministic" ), SinkMode.REPLACE ); Flow flow = getPlatform().getFlowConnector().connect( "trap test", source, sink, trap, pipe ); flow.complete(); validateLength( flow, 6, null ); validateLength( flow.openTrap(), 4 ); } @Test public void testTrapEachEveryAllChained() throws Exception { getPlatform().copyFromLocal( inputFileApache ); Tap source = getPlatform().getTextFile( inputFileApache ); Pipe pipe = new Pipe( "map" ); pipe = new Each( pipe, new Fields( "line" ), new RegexParser( new Fields( "ip" ), "^[^ ]*" ), new Fields( "ip" ) ); pipe = new Each( pipe, AssertionLevel.VALID, new AssertNotEquals( "75.185.76.245" ) ); pipe = new GroupBy( pipe, new Fields( "ip" ) ); pipe = new Each( pipe, AssertionLevel.VALID, new AssertNotEquals( "68.46.103.112" ) ); pipe = new GroupBy( pipe, new Fields( "ip" ) ); pipe = new Each( pipe, AssertionLevel.VALID, new AssertNotEquals( "76.197.151.0" ) ); pipe = new Each( pipe, AssertionLevel.VALID, new AssertNotEquals( "12.215.138.88" ) ); Tap sink = getPlatform().getTextFile( getOutputPath( "eacheverychain/tap" ), SinkMode.REPLACE ); Tap trap = getPlatform().getTextFile( getOutputPath( "eacheverychain/trap" ), SinkMode.REPLACE ); Flow flow = getPlatform().getFlowConnector().connect( "trap test", source, sink, trap, pipe ); flow.complete(); validateLength( flow, 6, null ); validateLength( flow.openTrap(), 4 ); } @Test public void testTrapToSequenceFile() throws Exception { getPlatform().copyFromLocal( inputFileApache ); Tap source = getPlatform().getTextFile( inputFileApache ); Pipe pipe = new Pipe( "map" ); pipe = new Each( pipe, new Fields( "line" ), new RegexParser( new Fields( "ip" ), "^[^ ]*" ), new Fields( "ip" ) ); pipe = new Each( pipe, new Fields( "ip" ), new TestFunction( new Fields( "test" ), null ), Fields.ALL ); pipe = new GroupBy( "reduce", pipe, new Fields( "ip" ) ); pipe = new Every( pipe, new Count(), new Fields( "ip", "count" ) ); Tap sink = getPlatform().getTextFile( getOutputPath( "seq/tap" ), SinkMode.REPLACE ); Tap trap = getPlatform().getTabDelimitedFile( new Fields( "ip" ), getOutputPath( "seq/trap" ), SinkMode.REPLACE ); Flow flow = getPlatform().getFlowConnector().connect( "trap test", source, sink, trap, pipe ); flow.complete(); validateLength( flow, 0, null ); validateLength( flow.openTrap(), 10 ); } @Test public void testTrapTapSourceSink() throws Exception { getPlatform().copyFromLocal( inputFileApache ); Scheme scheme = getPlatform().getTestFailScheme(); Tap source = getPlatform().getTap( scheme, inputFileApache, SinkMode.KEEP ); Pipe pipe = new Pipe( "map" ); pipe = new Each( pipe, new Fields( "line" ), new RegexParser( new Fields( "ip" ), "^[^ ]*" ), new Fields( "ip" ) ); pipe = new GroupBy( pipe, new Fields( "ip" ) ); pipe = new Every( pipe, new Count(), new Fields( "ip", "count" ) ); Tap sink = getPlatform().getTap( scheme, getOutputPath( "trapsourcesink/sink" ), SinkMode.REPLACE ); Tap trap = getPlatform().getTextFile( new Fields( "line" ), getOutputPath( "trapsourcesink/trap" ), SinkMode.REPLACE ); Map<Object, Object> properties = getProperties(); getPlatform().setNumMapTasks( properties, 1 ); getPlatform().setNumReduceTasks( properties, 1 ); getPlatform().setNumGatherPartitionTasks( properties, 1 ); Flow flow = getPlatform().getFlowConnector( properties ).connect( "trap test", source, sink, trap, pipe ); flow.complete(); validateLength( flow.openTapForRead( getPlatform().getTextFile( sink.getIdentifier() ) ), 7 ); validateLength( flow.openTrap(), 2, Pattern.compile( "bad data" ) ); } @Test public void testTrapNoOperation() throws Exception { getPlatform().copyFromLocal( testDelimitedProblematic ); Tap source = getPlatform().getDelimitedFile( new Fields( "id", "name" ).applyTypes( int.class, String.class ), ",", testDelimitedProblematic ); Tap sink = getPlatform().getDelimitedFile( new Fields( "id", "name" ).applyTypes( int.class, String.class ), ",", getOutputPath( getTestName() ), SinkMode.REPLACE ); Tap trap = getPlatform().getTextFile( getOutputPath( getTestName() + "_trap" ), SinkMode.REPLACE ); Pipe pipe = new Pipe( "copy" ); FlowDef flowDef = FlowDef.flowDef() .addSource( pipe, source ) .addTailSink( pipe, sink ) .addTrap( pipe, trap ); Flow flow = getPlatform().getFlowConnector().connect( flowDef ); flow.complete(); validateLength( flow.openTrap(), 1 ); } @Test public void testTrapDiagnostics() throws Exception { getPlatform().copyFromLocal( inputFileApache ); Tap source = getPlatform().getTextFile( inputFileApache ); Pipe pipe = new Pipe( "map" ); pipe = new Each( pipe, new Fields( "line" ), new RegexParser( new Fields( "ip" ), "^[^ ]*" ), new Fields( "ip" ) ); pipe = new Each( pipe, new Fields( "ip" ), new TestFunction( new Fields( "test" ), null ), Fields.ALL ); pipe = new GroupBy( "reduce", pipe, new Fields( "ip" ) ); pipe = new Every( pipe, new Count(), new Fields( "ip", "count" ) ); Tap sink = getPlatform().getTextFile( getOutputPath( "diag/tap" + NONDETERMINISTIC ), SinkMode.REPLACE ); Tap trap = getPlatform().getTabDelimitedFile( Fields.ALL, getOutputPath( "diag/trap" + NONDETERMINISTIC ), SinkMode.REPLACE ); Map<Object, Object> properties = getProperties(); properties = TrapProps.trapProps() .recordAllDiagnostics() .buildProperties( properties ); Flow flow = getPlatform().getFlowConnector( properties ).connect( "trap test", source, sink, trap, pipe ); flow.complete(); validateLength( flow, 0 ); validateLength( flow.openTrap(), 10, 4, Pattern.compile( ".*TrapPlatformTest.*" ) ); } @Test public void testTrapDiagnosticsLocalConfig() throws Exception { getPlatform().copyFromLocal( inputFileApache ); Tap source = getPlatform().getTextFile( inputFileApache ); Pipe pipe = new Pipe( "map" ); pipe = new Each( pipe, new Fields( "line" ), new RegexParser( new Fields( "ip" ), "^[^ ]*" ), new Fields( "ip" ) ); pipe = new Each( pipe, new Fields( "ip" ), new TestFunction( new Fields( "test" ), null ), Fields.ALL ); pipe = new GroupBy( "reduce", pipe, new Fields( "ip" ) ); pipe = new Every( pipe, new Count(), new Fields( "ip", "count" ) ); Tap sink = getPlatform().getTextFile( getOutputPath( "diagconfigdef/tap" + NONDETERMINISTIC ), SinkMode.REPLACE ); Tap trap = getPlatform().getTabDelimitedFile( Fields.ALL, getOutputPath( "diagconfigdef/trap" + NONDETERMINISTIC ), SinkMode.REPLACE ); Map<Object, Object> properties = getProperties(); TrapProps.trapProps() .recordAllDiagnostics() .setProperties( trap.getConfigDef(), ConfigDef.Mode.DEFAULT ); Flow flow = getPlatform().getFlowConnector( properties ).connect( "trap test", source, sink, trap, pipe ); flow.complete(); validateLength( flow, 0 ); validateLength( flow.openTrap(), 10, 4, Pattern.compile( ".*TrapPlatformTest.*" ) ); } @Test(expected = CascadingException.class) public void testTrapFailure() throws Exception { getPlatform().copyFromLocal( inputFileApache ); Tap source = getPlatform().getTextFile( inputFileApache ); Scheme scheme = getPlatform().getTestFailScheme(); Tap trap2 = getPlatform().getTap( scheme, getOutputPath( "trapFailure/badTrap" ), SinkMode.REPLACE ); Tap sink = getPlatform().getTextFile( getOutputPath( "trapFailure/tap" ), SinkMode.REPLACE ); Pipe pipe = new Each( new Pipe( "firstPipe" ), new Fields( "line" ), new RegexParser( new Fields( "ip" ), "^[^ ]*" ), new Fields( "ip" ) ); pipe = new Each( new Pipe( "secondPipe", pipe ), new Fields( "ip" ), new TestFunction( new Fields( "test" ), null ), Fields.ALL ); Tap trap1 = getPlatform().getTextFile( getOutputPath( "trapFailure/firstTrap" ), SinkMode.REPLACE ); FlowDef flowDef = FlowDef.flowDef() .addSource( "firstPipe", source ) .addTrap( "firstPipe", trap1 ) .addTrap( "secondPipe", trap2 ) .addTail( pipe ) .addSink( pipe, sink ); Flow flow = getPlatform().getFlowConnector().connect( flowDef ); flow.complete(); } }