public class FlowPlatformTest extends PlatformTestCase { private static final Logger LOG = LoggerFactory.getLogger( FlowPlatformTest.class ); public FlowPlatformTest() { super( true ); } @Test public void testLocalModeSource() throws Exception { Tap source = new Lfs( new TextLine(), "input/path" ); Tap sink = new Hfs( new TextLine(), "output/path", SinkMode.REPLACE ); Pipe pipe = new Pipe( "test" ); Flow flow = getPlatform().getFlowConnector().connect( source, sink, pipe ); List<FlowStep> steps = flow.getFlowSteps(); assertEquals( "wrong size", 1, steps.size() ); FlowStep step = steps.get( 0 ); boolean isLocal = HadoopUtil.isLocal( (Configuration) step.getConfig() ); assertTrue( "is not local", isLocal ); } @Test public void testLocalModeSink() throws Exception { Tap source = new Hfs( new TextLine(), "input/path" ); Tap sink = new Lfs( new TextLine(), "output/path", SinkMode.REPLACE ); Pipe pipe = new Pipe( "test" ); Flow flow = getPlatform().getFlowConnector().connect( source, sink, pipe ); List<FlowStep> steps = flow.getFlowSteps(); assertEquals( "wrong size", 1, steps.size() ); FlowStep step = steps.get( 0 ); boolean isLocal = HadoopUtil.isLocal( (Configuration) step.getConfig() ); assertTrue( "is not local", isLocal ); } @Test public void testNotLocalMode() throws Exception { if( !getPlatform().isUseCluster() ) return; Tap source = new Hfs( new TextLine(), "input/path" ); Tap sink = new Hfs( new TextLine(), "output/path", SinkMode.REPLACE ); Pipe pipe = new Pipe( "test" ); Flow flow = getPlatform().getFlowConnector().connect( source, sink, pipe ); List<FlowStep> steps = flow.getFlowSteps(); assertEquals( "wrong size", 1, steps.size() ); FlowStep step = steps.get( 0 ); boolean isLocal = HadoopUtil.isLocal( (Configuration) ( (BaseFlowStep) step ).createInitializedConfig( flow.getFlowProcess(), ( (BaseHadoopPlatform) getPlatform() ).getConfiguration() ) ); assertTrue( "is local", !isLocal ); } @Test public void testStop() throws Exception { if( !getPlatform().isUseCluster() ) return; getPlatform().copyFromLocal( inputFileLower ); getPlatform().copyFromLocal( inputFileUpper ); Tap sourceLower = new Hfs( new TextLine( new Fields( "offset", "line" ) ), inputFileLower ); Tap sourceUpper = new Hfs( new TextLine( new Fields( "offset", "line" ) ), inputFileUpper ); Map sources = new HashMap(); sources.put( "lower", sourceLower ); sources.put( "upper", sourceUpper ); Function splitter = new RegexSplitter( new Fields( "num", "char" ), " " ); Tap sink = new Hfs( new TextLine(), getOutputPath( "stopped" ), SinkMode.REPLACE ); Pipe pipeLower = new Each( new Pipe( "lower" ), new Fields( "line" ), splitter ); pipeLower = new GroupBy( pipeLower, new Fields( "num" ) ); Pipe pipeUpper = new Each( new Pipe( "upper" ), new Fields( "line" ), splitter ); pipeUpper = new GroupBy( pipeUpper, new Fields( "num" ) ); Pipe splice = new CoGroup( pipeLower, new Fields( "num" ), pipeUpper, new Fields( "num" ), Fields.size( 4 ) ); final Flow flow = getPlatform().getFlowConnector( getProperties() ).connect( sources, sink, splice ); final LockingFlowListener listener = new LockingFlowListener(); flow.addListener( listener ); LOG.info( "calling start" ); flow.start(); Util.safeSleep( 5000 ); assertTrue( "did not start", listener.started.tryAcquire( 60, TimeUnit.SECONDS ) ); while( true ) { LOG.info( "testing if running" ); Thread.sleep( 1000 ); Map<String, FlowStepJob> map = Flows.getJobsMap( flow ); if( map == null || map.values().size() == 0 ) continue; FlowStepJob flowStepJob = map.values().iterator().next(); if( flowStepJob.getStepStats().getStatus() == CascadingStats.Status.FAILED ) fail( "failed to start Hadoop step, please check your environment." ); if( flowStepJob.isStarted() ) break; } final Semaphore start = new Semaphore( 0 ); final long startTime = System.nanoTime(); Future<Long> future = newSingleThreadExecutor().submit( () -> { start.release(); LOG.info( "calling complete" ); flow.complete(); return System.nanoTime() - startTime; } ); start.acquire(); LOG.info( "calling stop" ); flow.stop(); long stopTime = System.nanoTime() - startTime; long completeTime = future.get(); assertTrue( String.format( "stop: %s complete: %s", stopTime, completeTime ), stopTime <= completeTime ); assertTrue( "did not stop", listener.stopped.tryAcquire( 60, TimeUnit.SECONDS ) ); assertTrue( "did not complete", listener.completed.tryAcquire( 60, TimeUnit.SECONDS ) ); } private static class BadFilter extends BaseOperation implements Filter { private Object object = new Object(); public boolean isRemove( FlowProcess flowProcess, FilterCall filterCall ) { return false; } } @Test public void testFailedSerialization() throws Exception { getPlatform().copyFromLocal( inputFileLower ); Tap sourceLower = new Hfs( new TextLine( new Fields( "offset", "line" ) ), inputFileLower ); Map sources = new HashMap(); sources.put( "lower", sourceLower ); Function splitter = new RegexSplitter( new Fields( "num", "char" ), " " ); Tap sink = new Hfs( new TextLine(), getOutputPath( "badserialization" ), SinkMode.REPLACE ); Pipe pipeLower = new Each( new Pipe( "lower" ), new Fields( "line" ), splitter ); pipeLower = new Each( pipeLower, new Fields( "num" ), new BadFilter() ); pipeLower = new GroupBy( pipeLower, new Fields( "num" ) ); try { Flow flow = getPlatform().getFlowConnector( getProperties() ).connect( sources, sink, pipeLower ); fail( "did not throw serialization exception" ); } catch( Exception exception ) { } } @Test public void testStartStopRace() throws Exception { getPlatform().copyFromLocal( inputFileLower ); Tap sourceLower = new Hfs( new TextLine( new Fields( "offset", "line" ) ), inputFileLower ); Map sources = new HashMap(); sources.put( "lower", sourceLower ); Function splitter = new RegexSplitter( new Fields( "num", "char" ), " " ); Tap sink = new Hfs( new TextLine(), getOutputPath( "startstop" ), SinkMode.REPLACE ); Pipe pipeLower = new Each( new Pipe( "lower" ), new Fields( "line" ), splitter ); pipeLower = new GroupBy( pipeLower, new Fields( "num" ) ); Flow flow = getPlatform().getFlowConnector( getProperties() ).connect( sources, sink, pipeLower ); flow.start(); flow.stop(); } @Test public void testFailingListenerStarting() throws Exception { failingListenerTest( FailingFlowListener.OnFail.STARTING ); } @Test public void testFailingListenerStopping() throws Exception { failingListenerTest( FailingFlowListener.OnFail.STOPPING ); } @Test public void testFailingListenerCompleted() throws Exception { failingListenerTest( FailingFlowListener.OnFail.COMPLETED ); } @Test public void testFailingListenerThrowable() throws Exception { failingListenerTest( FailingFlowListener.OnFail.THROWABLE ); } private void failingListenerTest( FailingFlowListener.OnFail onFail ) throws Exception { getPlatform().copyFromLocal( inputFileLower ); getPlatform().copyFromLocal( inputFileUpper ); Tap sourceLower = new Hfs( new TextLine( new Fields( "offset", "line" ) ), inputFileLower ); Tap sourceUpper = new Hfs( new TextLine( new Fields( "offset", "line" ) ), inputFileUpper ); Map sources = new HashMap(); sources.put( "lower", sourceLower ); sources.put( "upper", sourceUpper ); Function splitter = new RegexSplitter( new Fields( "num", "char" ), " " ); Tap sink = new Hfs( new TextLine(), getOutputPath( onFail + "/stopped" ), SinkMode.REPLACE ); Pipe pipeLower = new Each( new Pipe( "lower" ), new Fields( "line" ), splitter ); if( onFail == FailingFlowListener.OnFail.THROWABLE ) { pipeLower = new Each( pipeLower, new Debug() { @Override public boolean isRemove( FlowProcess flowProcess, FilterCall filterCall ) { throw new RuntimeException( "failing inside pipe assembly intentionally" ); } } ); } pipeLower = new GroupBy( pipeLower, new Fields( "num" ) ); Pipe pipeUpper = new Each( new Pipe( "upper" ), new Fields( "line" ), splitter ); pipeUpper = new GroupBy( pipeUpper, new Fields( "num" ) ); Pipe splice = new CoGroup( pipeLower, new Fields( "num" ), pipeUpper, new Fields( "num" ), Fields.size( 4 ) ); Flow flow = getPlatform().getFlowConnector( getProperties() ).connect( sources, sink, splice ); FailingFlowListener listener = new FailingFlowListener( onFail ); flow.addListener( listener ); LOG.info( "calling start" ); flow.start(); assertTrue( "did not start", listener.started.tryAcquire( 120, TimeUnit.SECONDS ) ); if( onFail == FailingFlowListener.OnFail.STOPPING ) { while( true ) { LOG.info( "testing if running" ); Thread.sleep( 1000 ); Map<String, FlowStepJob> map = Flows.getJobsMap( flow ); if( map == null || map.values().size() == 0 ) continue; FlowStepJob flowStepJob = map.values().iterator().next(); if( flowStepJob.getStepStats().getStatus() == CascadingStats.Status.FAILED ) fail( "failed to start Hadoop step, please check your environment." ); if( flowStepJob.isStarted() ) break; } LOG.info( "calling stop" ); flow.stop(); } assertTrue( "did not complete", listener.completed.tryAcquire( 360, TimeUnit.SECONDS ) ); assertTrue( "did not stop", listener.stopped.tryAcquire( 360, TimeUnit.SECONDS ) ); try { flow.complete(); fail( "did not rethrow exception from listener" ); } catch( Exception exception ) { } } @Test public void testFlowID() throws Exception { Tap source = new Lfs( new TextLine(), "input/path" ); Tap sink = new Hfs( new TextLine(), "output/path", SinkMode.REPLACE ); Pipe pipe = new Pipe( "test" ); Map<Object, Object> props = getProperties(); Flow flow1 = getPlatform().getFlowConnector( props ).connect( source, sink, pipe ); assertNotNull( "missing id", flow1.getID() ); assertNotNull( "missing id in conf", flow1.getProperty( "cascading.flow.id" ) ); Flow flow2 = getPlatform().getFlowConnector( props ).connect( source, sink, pipe ); assertTrue( "same id", !flow1.getID().equalsIgnoreCase( flow2.getID() ) ); } @Test public void testCopyConfig() throws Exception { Tap source = new Lfs( new TextLine(), "input/path" ); Tap sink = new Hfs( new TextLine(), "output/path", SinkMode.REPLACE ); Pipe pipe = new Pipe( "test" ); Configuration conf = ( (BaseHadoopPlatform) getPlatform() ).getConfiguration(); conf.set( AppProps.APP_NAME, "testname" ); AppProps props = AppProps.appProps().setVersion( "1.2.3" ); Properties properties = props.buildProperties( conf ); Flow flow = getPlatform().getFlowConnector( properties ).connect( source, sink, pipe ); assertEquals( "testname", flow.getProperty( AppProps.APP_NAME ) ); assertEquals( "1.2.3", flow.getProperty( AppProps.APP_VERSION ) ); } @Test public void testStartWithoutComplete() throws Exception { getPlatform().copyFromLocal( inputFileLower ); Tap sourceLower = new Hfs( new TextLine( new Fields( "offset", "line" ) ), inputFileLower ); Map sources = new HashMap(); sources.put( "lower", sourceLower ); Function splitter = new RegexSplitter( new Fields( "num", "char" ), " " ); Tap sink = new Hfs( new TextLine(), getOutputPath( "withoutcomplete" ), SinkMode.REPLACE ); Pipe pipeLower = new Each( new Pipe( "lower" ), new Fields( "line" ), splitter ); pipeLower = new GroupBy( pipeLower, new Fields( "num" ) ); Flow flow = getPlatform().getFlowConnector( getProperties() ).connect( sources, sink, pipeLower ); LockingFlowListener listener = new LockingFlowListener(); flow.addListener( listener ); flow.start(); assertTrue( listener.completed.tryAcquire( 90, TimeUnit.SECONDS ) ); } @Test public void testFailOnMissingSuccessFlowListener() throws Exception { getPlatform().copyFromLocal( inputFileLower ); FlowListener listener = new FailOnMissingSuccessFlowListener(); Hfs source = new Hfs( new TextLine( new Fields( "offset", "line" ) ), inputFileLower ); Hfs success = new Hfs( new TextLine(), getOutputPath( "withsuccess" ), SinkMode.REPLACE ); Hfs without = new Hfs( new TextLine(), getOutputPath( "withoutsuccess" ), SinkMode.REPLACE ); Hfs sink = new Hfs( new TextLine(), getOutputPath( "final" ), SinkMode.REPLACE ); Flow firstFlow = getPlatform().getFlowConnector( getProperties() ).connect( source, success, new Pipe( "lower" ) ); firstFlow.addListener( listener ); firstFlow.complete(); Flow secondFlow = getPlatform().getFlowConnector( getProperties() ).connect( success, without, new Pipe( "lower" ) ); secondFlow.addListener( listener ); secondFlow.complete(); Hfs successTap = new Hfs( new TextLine(), new Path( without.getPath(), "_SUCCESS" ).toString() ); assertTrue( successTap.deleteResource( getPlatform().getFlowProcess() ) ); Flow thirdFlow = getPlatform().getFlowConnector( getProperties() ).connect( without, sink, new Pipe( "lower" ) ); thirdFlow.addListener( listener ); try { thirdFlow.complete(); fail( "listener did not fail flow" ); } catch( FlowException exception ) { } } }