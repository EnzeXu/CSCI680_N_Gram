public class AppProps extends Props { private static final Logger LOG = LoggerFactory . getLogger ( AppProps . class ) ; public static final String APP_ID = "cascading . app . id" ; public static final String APP_NAME = "cascading . app . name" ; public static final String APP_VERSION = "cascading . app . version" ; public static final String APP_TAGS = "cascading . app . tags" ; public static final String APP_FRAMEWORKS = "cascading . app . frameworks" ; public static final String APP_JAR_CLASS = "cascading . app . appjar . class" ; public static final String APP_JAR_PATH = "cascading . app . appjar . path" ; static final String DEP_APP_JAR_CLASS = "cascading . flowconnector . appjar . class" ; static final String DEP_APP_JAR_PATH = "cascading . flowconnector . appjar . path" ; private static String appID ; protected String name ; protected String version ; protected Set < String > tags = new TreeSet < String > ( ) ; protected Class jarClass ; protected String jarPath ; protected Set < String > frameworks = new TreeSet < String > ( ) ; public static AppProps appProps ( ) { return new AppProps ( ) ; } public static void setApplicationJarClass ( Map < Object , Object > properties , Class type ) { if ( type != null ) PropertyUtil . setProperty ( properties , APP_JAR_CLASS , type . getName ( ) ) ; } public static Class < ? > getApplicationJarClass ( Map < Object , Object > properties ) { Object type = PropertyUtil . getProperty ( properties , DEP_APP_JAR_CLASS , null ) ; if ( type instanceof Class ) { LOG . warn ( "using deprecated property : { } , use instead : { } " , DEP_APP_JAR_CLASS , APP_JAR_CLASS ) ; return ( Class < ? > ) type ; } String className = ( String ) type ; if ( className != null ) { LOG . warn ( "using deprecated property : { } , use instead : { } " , DEP_APP_JAR_CLASS , APP_JAR_CLASS ) ; return Util . loadClassSafe ( className ) ; } type = PropertyUtil . getProperty ( properties , APP_JAR_CLASS , null ) ; if ( type instanceof Class ) return ( Class < ? > ) type ; className = ( String ) type ; if ( className == null ) return null ; return Util . loadClassSafe ( className ) ; } public static void setApplicationJarPath ( Map < Object , Object > properties , String path ) { if ( path != null ) properties . put ( APP_JAR_PATH , path ) ; } public static String getApplicationJarPath ( Map < Object , Object > properties ) { String property = PropertyUtil . getProperty ( properties , DEP_APP_JAR_PATH , ( String ) null ) ; if ( property != null ) { LOG . warn ( "using deprecated property : { } , use instead : { } " , DEP_APP_JAR_PATH , APP_JAR_PATH ) ; return property ; } return PropertyUtil . getProperty ( properties , APP_JAR_PATH , ( String ) null ) ; } public static void setApplicationID ( Map < Object , Object > properties ) { properties . put ( APP_ID , getAppID ( ) ) ; } public static String getApplicationID ( Map < Object , Object > properties ) { if ( properties == null ) return getAppID ( ) ; return PropertyUtil . getProperty ( properties , APP_ID , getAppID ( ) ) ; } public static String getApplicationID ( ) { return getAppID ( ) ; } private static String getAppID ( ) { if ( appID == null ) { appID = Util . createUniqueID ( ) ; LOG . info ( "using app . id : { } " , appID ) ; } return appID ; } public static void resetAppID ( ) { appID = null ; } public static void setApplicationName ( Map < Object , Object > properties , String name ) { if ( name != null ) properties . put ( APP_NAME , name ) ; } public static String getApplicationName ( Map < Object , Object > properties ) { return PropertyUtil . getProperty ( properties , APP_NAME , ( String ) null ) ; } public static void setApplicationVersion ( Map < Object , Object > properties , String version ) { if ( version != null ) properties . put ( APP_VERSION , version ) ; } public static String getApplicationVersion ( Map < Object , Object > properties ) { return PropertyUtil . getProperty ( properties , APP_VERSION , ( String ) null ) ; } public static void addApplicationTag ( Map < Object , Object > properties , String tag ) { if ( tag == null ) return ; tag = tag . trim ( ) ; if ( Util . containsWhitespace ( tag ) ) LOG . warn ( "tags should not contain whitespace characters : ' { } '" , tag ) ; String tags = PropertyUtil . getProperty ( properties , APP_TAGS , ( String ) null ) ; if ( tags != null ) tags = join ( " , " , tag , tags ) ; else tags = tag ; properties . put ( APP_TAGS , tags ) ; } public static String getApplicationTags ( Map < Object , Object > properties ) { return PropertyUtil . getProperty ( properties , APP_TAGS , ( String ) null ) ; } public static void addApplicationFramework ( Map < Object , Object > properties , String framework ) { if ( framework == null ) return ; String frameworks = PropertyUtil . getProperty ( properties , APP_FRAMEWORKS , System . getProperty ( APP_FRAMEWORKS ) ) ; if ( frameworks != null ) frameworks = join ( " , " , framework . trim ( ) , frameworks ) ; else frameworks = framework ; frameworks = Util . unique ( frameworks , " , " ) ; if ( properties != null ) properties . put ( APP_FRAMEWORKS , frameworks ) ; System . setProperty ( APP_FRAMEWORKS , frameworks ) ; } public static String getApplicationFrameworks ( Map < Object , Object > properties ) { return PropertyUtil . getProperty ( properties , APP_FRAMEWORKS , System . getProperty ( APP_FRAMEWORKS ) ) ; } public AppProps ( ) { } public AppProps ( String name , String version ) { this . name = name ; this . version = version ; } public AppProps setName ( String name ) { this . name = name ; return this ; } public AppProps setVersion ( String version ) { this . version = version ; return this ; } public String getTags ( ) { return join ( tags , " , " ) ; } public AppProps addTag ( String tag ) { if ( !Util . isEmpty ( tag ) ) tags . add ( tag ) ; return this ; } public AppProps addTags ( String . . . tags ) { for ( String tag : tags ) addTag ( tag ) ; return this ; } public String getFrameworks ( ) { return join ( frameworks , " , " ) ; } public AppProps addFramework ( String framework ) { if ( !Util . isEmpty ( framework ) ) frameworks . add ( framework ) ; return this ; } public AppProps addFramework ( String framework , String version ) { if ( !Util . isEmpty ( framework ) && !Util . isEmpty ( version ) ) frameworks . add ( framework + " : " + version ) ; if ( !Util . isEmpty ( framework ) ) frameworks . add ( framework ) ; return this ; } public AppProps addFrameworks ( String . . . frameworks ) { for ( String framework : frameworks ) addFramework ( framework ) ; return this ; } public AppProps setJarClass ( Class jarClass ) { this . jarClass = jarClass ; return this ; } public AppProps setJarPath ( String jarPath ) { this . jarPath = jarPath ; return this ; } @ Override protected void addPropertiesTo ( Properties properties ) { setApplicationID ( properties ) ; setApplicationName ( properties , name ) ; setApplicationVersion ( properties , version ) ; addApplicationTag ( properties , getTags ( ) ) ; addApplicationFramework ( properties , getFrameworks ( ) ) ; setApplicationJarClass ( properties , jarClass ) ; setApplicationJarPath ( properties , jarPath ) ; } }