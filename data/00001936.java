public class EveryPipeAssemblyPlatformTest extends PipeAssemblyTestBase { public static Test suite( TestPlatform testPlatform ) throws Exception { TestSuite suite = new TestSuite(); Properties properties = loadProperties( "op.properties" ); makeSuites( testPlatform, properties, buildOpPipes( null, new Pipe( "every" ), new EveryAssemblyFactory(), OP_ARGS_FIELDS, OP_DECL_FIELDS, OP_SELECT_FIELDS, OP_VALUE, runOnly( properties ) ), suite, EveryPipeAssemblyPlatformTest.class ); return suite; } public EveryPipeAssemblyPlatformTest( Properties properties, String displayName, String name, Pipe pipe ) { super( properties, displayName, name, pipe ); } }