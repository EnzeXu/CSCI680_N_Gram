public class ResolverExceptionsPlatformTest extends PlatformTestCase { public ResolverExceptionsPlatformTest() { } private void verify( Tap source, Tap sink, Pipe pipe ) { try { getPlatform().getFlowConnector().connect( source, sink, pipe ); fail( "no exception thrown" ); } catch( Exception exception ) { assertTrue( exception instanceof PlannerException ); assertTrue( exception.getCause().getCause() instanceof FieldsResolverException ); } } @Test public void testSchemeResolver() throws Exception { Fields sourceFields = new Fields( "first", "second" ); Tap source = getPlatform().getTabDelimitedFile( sourceFields, "input/path", SinkMode.KEEP ); Fields sinkFields = new Fields( "third", "fourth" ); Tap sink = getPlatform().getTabDelimitedFile( sinkFields, "output/path", SinkMode.REPLACE ); Pipe pipe = new Pipe( "test" ); verify( source, sink, pipe ); } @Test public void testEachArgResolver() throws Exception { Fields sourceFields = new Fields( "first", "second" ); Tap source = getPlatform().getTabDelimitedFile( sourceFields, "input/path", SinkMode.KEEP ); Fields sinkFields = new Fields( "third", "fourth" ); Tap sink = getPlatform().getTabDelimitedFile( sinkFields, "output/path", SinkMode.REPLACE ); Pipe pipe = new Pipe( "test" ); pipe = new Each( pipe, new Fields( "third" ), new Identity() ); verify( source, sink, pipe ); } @Test public void testEachOutResolver() throws Exception { Fields sourceFields = new Fields( "first", "second" ); Tap source = getPlatform().getTabDelimitedFile( sourceFields, "input/path", SinkMode.KEEP ); Fields sinkFields = new Fields( "third", "fourth" ); Tap sink = getPlatform().getTabDelimitedFile( sinkFields, "output/path", SinkMode.REPLACE ); Pipe pipe = new Pipe( "test" ); pipe = new Each( pipe, new Fields( "first" ), new Identity( new Fields( "none" ) ), new Fields( "third" ) ); verify( source, sink, pipe ); } @Test public void testGroupByResolver() throws Exception { Fields sourceFields = new Fields( "first", "second" ); Tap source = getPlatform().getTabDelimitedFile( sourceFields, "input/path", SinkMode.KEEP ); Fields sinkFields = new Fields( "third", "fourth" ); Tap sink = getPlatform().getTabDelimitedFile( sinkFields, "output/path", SinkMode.REPLACE ); Pipe pipe = new Pipe( "test" ); pipe = new GroupBy( pipe, new Fields( "third" ) ); verify( source, sink, pipe ); } @Test public void testGroupBySortResolver() throws Exception { Fields sourceFields = new Fields( "first", "second" ); Tap source = getPlatform().getTabDelimitedFile( sourceFields, "input/path", SinkMode.KEEP ); Fields sinkFields = new Fields( "third", "fourth" ); Tap sink = getPlatform().getTabDelimitedFile( sinkFields, "output/path", SinkMode.REPLACE ); Pipe pipe = new Pipe( "test" ); pipe = new GroupBy( pipe, new Fields( "first" ), new Fields( "third" ) ); verify( source, sink, pipe ); } @Test public void testEveryArgResolver() throws Exception { Fields sourceFields = new Fields( "first", "second" ); Tap source = getPlatform().getTabDelimitedFile( sourceFields, "input/path", SinkMode.KEEP ); Fields sinkFields = new Fields( "third", "fourth" ); Tap sink = getPlatform().getTabDelimitedFile( sinkFields, "output/path", SinkMode.REPLACE ); Pipe pipe = new Pipe( "test" ); pipe = new GroupBy( pipe, new Fields( "first" ) ); pipe = new Every( pipe, new Fields( "third" ), new Count() ); verify( source, sink, pipe ); } @Test public void testEveryOutResolver() throws Exception { Fields sourceFields = new Fields( "first", "second" ); Tap source = getPlatform().getTabDelimitedFile( sourceFields, "input/path", SinkMode.KEEP ); Fields sinkFields = new Fields( "third", "fourth" ); Tap sink = getPlatform().getTabDelimitedFile( sinkFields, "output/path", SinkMode.REPLACE ); Pipe pipe = new Pipe( "test" ); pipe = new GroupBy( pipe, new Fields( "first" ) ); pipe = new Every( pipe, new Fields( "second" ), new Count(), new Fields( "third" ) ); verify( source, sink, pipe ); } }