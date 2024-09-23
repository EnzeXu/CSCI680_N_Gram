public class JavaLocator { private static final boolean IS_WINDOWS = System . getProperty ( "os . name" ) . toLowerCase ( Locale . ROOT ) . startsWith ( "windows" ) ; public static File findExecutableFromToolchain ( Toolchain toolchain ) { if ( toolchain != null ) { String fromToolChain = toolchain . findTool ( "java" ) ; if ( fromToolChain != null ) { return new File ( fromToolChain ) ; } } String javaCommand = "java" + ( IS_WINDOWS ? " . exe" : "" ) ; String javaHomeSystemProperty = System . getProperty ( "java . home" ) ; if ( javaHomeSystemProperty != null ) { Path javaHomePath = Paths . get ( javaHomeSystemProperty ) ; if ( javaHomePath . endsWith ( "jre" ) ) { File javaExecPath = javaHomePath . resolveSibling ( "bin" ) . resolve ( javaCommand ) . toFile ( ) ; if ( javaExecPath . isFile ( ) ) { return javaExecPath ; } } File javaExecPath = javaHomePath . resolve ( "bin" ) . resolve ( javaCommand ) . toFile ( ) ; if ( javaExecPath . isFile ( ) ) { return javaExecPath ; } else { throw new IllegalStateException ( "Couldn't locate java in defined java . home system property . " ) ; } } String javaHomeEnvVar = System . getenv ( "JAVA_HOME" ) ; if ( javaHomeEnvVar == null ) { throw new IllegalStateException ( "Couldn't locate java , try setting JAVA_HOME environment variable . " ) ; } File javaExecPath = Paths . get ( javaHomeEnvVar ) . resolve ( "bin" ) . resolve ( javaCommand ) . toFile ( ) ; if ( javaExecPath . isFile ( ) ) { return javaExecPath ; } else { throw new IllegalStateException ( "Couldn't locate java in defined JAVA_HOME environment variable . " ) ; } } public static File findHomeFromToolchain ( Toolchain toolchain ) { File executable = findExecutableFromToolchain ( toolchain ) ; File executableParent = executable . getParentFile ( ) ; if ( executableParent == null ) { return null ; } return executableParent . getParentFile ( ) ; } }