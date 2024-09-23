public class ScalaRunMojo extends ScalaMojoSupport { @ Parameter ( property = "launcher" ) private String launcher ; @ Parameter ( property = "addArgs" ) private String addArgs ; @ Parameter private Launcher [ ] launchers ; @ Parameter ( property = "mainClass" ) private String mainClass ; @ Override protected void doExecute ( ) throws Exception { JavaMainCaller jcmd = null ; File javaExec = JavaLocator . findExecutableFromToolchain ( getToolchain ( ) ) ; if ( StringUtils . isNotEmpty ( mainClass ) ) { jcmd = new JavaMainCallerByFork ( getLog ( ) , mainClass , FileUtils . toMultiPath ( FileUtils . fromStrings ( project . getTestClasspathElements ( ) ) ) , jvmArgs , args , forceUseArgFile , javaExec ) ; } else if ( ( launchers != null ) && ( launchers . length > 0 ) ) { if ( StringUtils . isNotEmpty ( launcher ) ) { for ( int i = 0 ; ( i < launchers . length ) && ( jcmd == null ) ; i++ ) { if ( launcher . equals ( launchers [ i ] . id ) ) { getLog ( ) . info ( "launcher '" + launchers [ i ] . id + "' selected = > " + launchers [ i ] . mainClass ) ; jcmd = new JavaMainCallerByFork ( getLog ( ) , launchers [ i ] . mainClass , FileUtils . toMultiPath ( FileUtils . fromStrings ( project . getTestClasspathElements ( ) ) ) , launchers [ i ] . jvmArgs , launchers [ i ] . args , forceUseArgFile , javaExec ) ; } } } else { getLog ( ) . info ( "launcher '" + launchers [ 0 ] . id + "' selected = > " + launchers [ 0 ] . mainClass ) ; jcmd = new JavaMainCallerByFork ( getLog ( ) , launchers [ 0 ] . mainClass , FileUtils . toMultiPath ( FileUtils . fromStrings ( project . getTestClasspathElements ( ) ) ) , launchers [ 0 ] . jvmArgs , launchers [ 0 ] . args , forceUseArgFile , javaExec ) ; } } if ( jcmd != null ) { if ( StringUtils . isNotEmpty ( addArgs ) ) { jcmd . addArgs ( StringUtils . split ( addArgs , "|" ) ) ; } jcmd . run ( displayCmd ) ; } else { getLog ( ) . warn ( "Not mainClass or valid launcher found/define" ) ; } } }