public class PlatformRunner extends ParentRunner<Runner> { public static final String PLATFORM_INCLUDES = "test.platform.includes"; public static final String PLATFORM_RESOURCE = "cascading/platform/platform.properties"; public static final String PLATFORM_CLASSNAME = "platform.classname"; private static final Logger LOG = LoggerFactory.getLogger( PlatformRunner.class ); private Set<String> includes = new HashSet<String>(); private List<Runner> runners; @Retention(RetentionPolicy.RUNTIME) @Inherited public @interface Platform { Class<? extends TestPlatform>[] value(); } public PlatformRunner( Class<PlatformTestCase> testClass ) throws Throwable { super( testClass ); setIncludes(); makeRunners(); } private void setIncludes() { String includesString = System.getProperty( PLATFORM_INCLUDES ); if( includesString == null || includesString.isEmpty() ) return; String[] split = includesString.split( "," ); for( String include : split ) includes.add( include.trim().toLowerCase() ); } public static TestPlatform makeInstance( Class<? extends TestPlatform> type ) { try { return type.newInstance(); } catch( NoClassDefFoundError | InstantiationException | IllegalAccessException exception ) { throw new RuntimeException( exception ); } } @Override protected List<Runner> getChildren() { return runners; } private List<Runner> makeRunners() throws Throwable { Class<?> javaClass = getTestClass().getJavaClass(); runners = new ArrayList<Runner>(); Set<Class<? extends TestPlatform>> classes = getPlatformClassesFromAnnotation( javaClass ); if( classes.isEmpty() ) classes = getPlatformClassesFromClasspath( javaClass.getClassLoader() ); int count = 0; Iterator<Class<? extends TestPlatform>> iterator = classes.iterator(); while( iterator.hasNext() ) addPlatform( javaClass, iterator.next(), count++, classes.size() ); return runners; } private Set<Class<? extends TestPlatform>> getPlatformClassesFromAnnotation( Class<?> javaClass ) throws Throwable { PlatformRunner.Platform annotation = javaClass.getAnnotation( PlatformRunner.Platform.class ); if( annotation == null ) return Collections.EMPTY_SET; HashSet<Class<? extends TestPlatform>> classes = new LinkedHashSet<Class<? extends TestPlatform>>( Arrays.asList( annotation.value() ) ); LOG.info( "found {} test platforms from Platform annotation", classes.size() ); return classes; } static Map<ClassLoader, Set<Class<? extends TestPlatform>>> cache = new WeakHashMap<>(); protected synchronized static Set<Class<? extends TestPlatform>> getPlatformClassesFromClasspath( ClassLoader classLoader ) throws IOException, ClassNotFoundException { if( cache.containsKey( classLoader ) ) return cache.get( classLoader ); Set<Class<? extends TestPlatform>> classes = new LinkedHashSet<>(); Properties properties = new Properties(); LOG.debug( "classloader: {}", classLoader ); Enumeration<URL> urls = classLoader.getResources( PLATFORM_RESOURCE ); while( urls.hasMoreElements() ) { InputStream stream = urls.nextElement().openStream(); classes.add( (Class<? extends TestPlatform>) getPlatformClass( classLoader, properties, stream ) ); } if( classes.isEmpty() ) { LOG.warn( "no platform tests will be run" ); LOG.warn( "did not find {} in the classpath, no {} instances found", PLATFORM_RESOURCE, TestPlatform.class.getCanonicalName() ); LOG.warn( "add cascading-local, cascading-hadoop, and/or external planner library to the test classpath" ); } else { LOG.info( "found {} test platforms from classpath", classes.size() ); } cache.put( classLoader, classes ); return classes; } private static Class<?> getPlatformClass( ClassLoader classLoader, Properties properties, InputStream stream ) throws IOException, ClassNotFoundException { if( stream == null ) throw new IllegalStateException( "platform provider resource not found: " + PLATFORM_RESOURCE ); properties.load( stream ); String classname = properties.getProperty( PLATFORM_CLASSNAME ); if( classname == null ) throw new IllegalStateException( "platform provider value not found: " + PLATFORM_CLASSNAME ); Class<?> type = classLoader.loadClass( classname ); if( type == null ) throw new IllegalStateException( "platform provider class not found: " + classname ); return type; } private void addPlatform( final Class<?> javaClass, Class<? extends TestPlatform> type, int ordinal, int size ) throws Throwable { if( javaClass.getAnnotation( Ignore.class ) != null ) { LOG.info( "ignoring test class: {}", javaClass.getCanonicalName() ); return; } final TestPlatform testPlatform = makeInstance( type ); if( testPlatform == null ) return; final String platformName = testPlatform.getName(); if( !includes.isEmpty() && !includes.contains( platformName.toLowerCase() ) ) { LOG.info( "ignoring platform: {}", platformName ); return; } LOG.info( "adding test: {}, with platform: {}", javaClass.getName(), platformName ); PlatformSuite suiteAnnotation = javaClass.getAnnotation( PlatformSuite.class ); if( suiteAnnotation != null ) runners.add( makeSuiteRunner( javaClass, suiteAnnotation.method(), testPlatform ) ); else runners.add( makeClassRunner( javaClass, testPlatform, platformName, size != 1 ) ); } private JUnit38ClassRunner makeSuiteRunner( Class<?> javaClass, String suiteMethod, final TestPlatform testPlatform ) throws Throwable { Method method = javaClass.getMethod( suiteMethod, TestPlatform.class ); return new JUnit38ClassRunner( (Test) method.invoke( null, testPlatform ) ); } private BlockJUnit4ClassRunner makeClassRunner( final Class<?> javaClass, final TestPlatform testPlatform, final String platformName, final boolean useName ) throws InitializationError { return new BlockJUnit4ClassRunner( javaClass ) { @Override protected String getName() { if( useName ) return String.format( "%s[%s]", super.getName(), platformName ); else return super.getName(); } @Override protected Object createTest() throws Exception { PlatformTestCase testCase = (PlatformTestCase) super.createTest(); testCase.installPlatform( testPlatform ); return testCase; } }; } @Override protected Description describeChild( Runner runner ) { return runner.getDescription(); } @Override protected void runChild( Runner runner, RunNotifier runNotifier ) { runner.run( runNotifier ); } @Override public void filter( Filter filter ) throws NoTestsRemainException { for( Runner runner : getChildren() ) { if( runner instanceof Filterable ) ( (Filterable) runner ).filter( filter ); } } }