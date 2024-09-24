public class TraceTest extends CascadingTestCase { @Test public void testOperation() { BaseOperation operation = new Identity(); assertEqualsTrace( "cascading.TraceTest.testOperation(TraceTest.java", operation.getTrace() ); } @Test public void testPipe() { Pipe pipe = new Pipe( "foo" ); assertEqualsTrace( "cascading.TraceTest.testPipe(TraceTest.java", pipe.getTrace() ); } @Test public void testPipeEach() { Pipe pipe = new Pipe( "foo" ); pipe = new Each( pipe, new Fields( "a" ), new Identity() ); assertEqualsTrace( "cascading.TraceTest.testPipeEach(TraceTest.java", pipe.getTrace() ); } @Test public void testPipeCoGroup() { Pipe pipe = new Pipe( "foo" ); pipe = new Each( pipe, new Fields( "a" ), new Identity() ); pipe = new CoGroup( pipe, new Fields( "b" ), 4 ); assertEqualsTrace( "cascading.TraceTest.testPipeCoGroup(TraceTest.java", pipe.getTrace() ); } @Test public void testPipeHashJoin() { Pipe pipe = new Pipe( "foo" ); pipe = new Each( pipe, new Fields( "a" ), new Identity() ); pipe = new HashJoin( pipe, new Fields( "b" ), new Pipe( "bar" ), new Fields( "c" ) ); assertEqualsTrace( "cascading.TraceTest.testPipeHashJoin(TraceTest.java", pipe.getTrace() ); } @Test public void testPipeGroupBy() { Pipe pipe = new Pipe( "foo" ); pipe = new Each( pipe, new Fields( "a" ), new Identity() ); pipe = new GroupBy( pipe, new Fields( "b" ) ); assertEqualsTrace( "cascading.TraceTest.testPipeGroupBy(TraceTest.java", pipe.getTrace() ); } @Test public void testPipeMerge() { Pipe pipe = new Pipe( "foo" ); pipe = new Each( pipe, new Fields( "a" ), new Identity() ); pipe = new Merge( pipe, new Pipe( "bar" ) ); assertEqualsTrace( "cascading.TraceTest.testPipeMerge(TraceTest.java", pipe.getTrace() ); } @Test public void testPipeAssembly() { Pipe pipe = new Pipe( "foo" ); pipe = new Rename( pipe, new Fields( "a" ), new Fields( "b" ) ); assertEqualsTrace( "cascading.TraceTest.testPipeAssembly(TraceTest.java", pipe.getTrace() ); } protected static class TestSubAssembly extends SubAssembly { public Pipe pipe; public TestSubAssembly() { Pipe pipe = new Pipe( "foo" ); setPrevious( pipe ); pipe = new Rename( pipe, new Fields( "a" ), new Fields( "b" ) ); this.pipe = pipe; setTails( pipe ); } } @Test public void testPipeAssemblyDeep() { TestSubAssembly pipe = new TestSubAssembly(); assertEqualsTrace( "cascading.TraceTest.testPipeAssemblyDeep(TraceTest.java", pipe.getTrace() ); assertEqualsTrace( "cascading.TraceTest$TestSubAssembly.<init>(TraceTest.java", pipe.pipe.getTrace() ); assertEqualsTrace( "cascading.TraceTest$TestSubAssembly.<init>(TraceTest.java", pipe.getTails()[ 0 ].getTrace() ); } public static Pipe sampleApi() { return new Pipe( "foo" ); } @Test public void testApiBoundary() { final String regex = "cascading\\.TraceTest\\.sampleApi.*"; TraceUtil.registerApiBoundary( regex ); try { Pipe pipe1 = sampleApi(); assertEqualsTrace( "sampleApi() @ cascading.TraceTest.testApiBoundary(TraceTest.java", pipe1.getTrace() ); } finally { TraceUtil.unregisterApiBoundary( regex ); } Pipe pipe2 = sampleApi(); assertEqualsTrace( "cascading.TraceTest.sampleApi(TraceTest.java", pipe2.getTrace() ); } @Test public void testTap() { Tap tap = new Tap() { @Override public String getIdentifier() { return null; } @Override public TupleEntryIterator openForRead( FlowProcess flowProcess, Object object ) throws IOException { return null; } @Override public TupleEntryCollector openForWrite( FlowProcess flowProcess, Object object ) throws IOException { return null; } @Override public boolean createResource( Object conf ) throws IOException { return false; } @Override public boolean deleteResource( Object conf ) throws IOException { return false; } @Override public boolean resourceExists( Object conf ) throws IOException { return false; } @Override public long getModifiedTime( Object conf ) throws IOException { return 0; } }; assertEqualsTrace( "cascading.TraceTest.testTap(TraceTest.java", tap.getTrace() ); } @Test public void testScheme() { Scheme scheme = new Scheme() { @Override public void sourceConfInit( FlowProcess flowProcess, Tap tap, Object conf ) { } @Override public void sinkConfInit( FlowProcess flowProcess, Tap tap, Object conf ) { } @Override public boolean source( FlowProcess flowProcess, SourceCall sourceCall ) throws IOException { return false; } @Override public void sink( FlowProcess flowProcess, SinkCall sinkCall ) throws IOException { } }; assertEqualsTrace( "cascading.TraceTest.testScheme(TraceTest.java", scheme.getTrace() ); } public static void assertEqualsTrace( String expected, String trace ) { String substring = trace.substring( 0, trace.lastIndexOf( ":" ) ); assertEquals( expected, substring ); } @Test public void testStringify() { Throwable top = new CascadingException( "test message", new NullPointerException( "had an npe" ) ); assertNotNull( TraceUtil.stringifyStackTrace( top, "|", true, -1 ) ); assertTrue( !TraceUtil.stringifyStackTrace( top, "|", true, -1 ).contains( "\t" ) ); assertTrue( TraceUtil.stringifyStackTrace( top, "|", false, -1 ).contains( "\t" ) ); assertNull( TraceUtil.stringifyStackTrace( top, "|", true, 0 ) ); assertEquals( 1, TraceUtil.stringifyStackTrace( top, "|", true, 1 ).split( "\\|" ).length ); assertEquals( 2, TraceUtil.stringifyStackTrace( top, "|", true, 2 ).split( "\\|" ).length ); } }