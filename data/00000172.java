public class ScalaConsoleMojo extends ScalaMojoSupport { private static final String JLINE = "jline" ; private static final String SCALA_ORG_GROUP = "org . scala-lang" ; @ Parameter ( property = "mainConsole" , defaultValue = "scala . tools . nsc . MainGenericRunner" , required = true ) private String mainConsole ; @ Parameter ( property = "maven . scala . console . useTestClasspath" , defaultValue = "true" , required = true ) private boolean useTestClasspath ; @ Parameter ( property = "maven . scala . console . useRuntimeClasspath" , defaultValue = "true" , required = true ) private boolean useRuntimeClasspath ; @ Parameter ( property = "javarebel . jar . path" ) private File javaRebelPath ; @ Override protected void doExecute ( ) throws Exception { Context sc = findScalaContext ( ) ; final JavaMainCaller jcmd = super . getScalaCommand ( false , sc . consoleMainClassName ( mainConsole ) ) ; final VersionNumber scalaVersion = super . findScalaContext ( ) . version ( ) ; final Set < File > classpath = this . setupClassPathForConsole ( scalaVersion ) ; if ( super . fork ) { super . getLog ( ) . info ( "Ignoring fork for console execution . " ) ; } final String classpathStr = FileUtils . toMultiPath ( classpath ) ; jcmd . addArgs ( super . args ) ; jcmd . addOption ( "-cp" , classpathStr ) ; super . addCompilerPluginOptions ( jcmd ) ; this . handleJavaRebel ( jcmd ) ; jcmd . run ( super . displayCmd ) ; } private void handleJavaRebel ( final JavaMainCaller jcmd ) throws IOException { if ( this . javaRebelPath != null ) { final String canonicalJavaRebelPath = this . javaRebelPath . getCanonicalPath ( ) ; if ( this . javaRebelPath . exists ( ) ) { jcmd . addJvmArgs ( "-noverify" , String . format ( "-javaagent : %s" , canonicalJavaRebelPath ) ) ; } else { super . getLog ( ) . warn ( String . format ( "javaRevelPath '%s' not found" , canonicalJavaRebelPath ) ) ; } } } private Set < File > setupClassPathForConsole ( final VersionNumber scalaVersion ) throws Exception { final Set < File > classpath = new TreeSet < File > ( ) ; classpath . addAll ( this . setupProjectClasspaths ( ) ) ; classpath . addAll ( this . setupConsoleClasspaths ( scalaVersion ) ) ; return classpath ; } private Set < File > setupProjectClasspaths ( ) throws Exception { final Set < File > classpath = new TreeSet < > ( ) ; super . addCompilerToClasspath ( classpath ) ; super . addLibraryToClasspath ( classpath ) ; if ( this . useTestClasspath ) { for ( String s : super . project . getTestClasspathElements ( ) ) { classpath . add ( new File ( s ) ) ; } } if ( this . useRuntimeClasspath ) { for ( String s : super . project . getRuntimeClasspathElements ( ) ) { classpath . add ( new File ( s ) ) ; } } return classpath ; } private Set < File > setupConsoleClasspaths ( final VersionNumber scalaVersion ) throws Exception { final Set < File > classpath = new TreeSet < File > ( ) ; Artifact a = this . resolveJLine ( this . fallbackJLine ( scalaVersion ) ) ; addToClasspath ( a . getGroupId ( ) , a . getArtifactId ( ) , a . getVersion ( ) , a . getClassifier ( ) , classpath , true ) ; return classpath ; } private Artifact resolveJLine ( final Artifact defaultFallback ) throws Exception { final Set < Artifact > compilerDeps = findScalaContext ( ) . findCompilerAndDependencies ( ) ; for ( final Artifact a : compilerDeps ) { if ( filterForJline ( a ) ) { return a ; } } getLog ( ) . warn ( "Unable to determine the required Jline dependency from the POM . Falling back to hard-coded defaults . " ) ; getLog ( ) . warn ( "If you get an InvocationTargetException , then this probably means we guessed the wrong version for JLine" ) ; super . getLog ( ) . warn ( String . format ( "Guessed JLine : %s" , defaultFallback . toString ( ) ) ) ; return defaultFallback ; } private boolean filterForJline ( final Artifact artifact ) { final String artifactId = artifact . getArtifactId ( ) ; final String groupId = artifact . getGroupId ( ) ; return artifactId . equals ( ScalaConsoleMojo . JLINE ) && groupId . equals ( ScalaConsoleMojo . JLINE ) ; } private Artifact fallbackJLine ( final VersionNumber scalaVersion ) { final VersionNumber scala2_12_0M4 = new VersionNumber ( "2 . 12 . 0-M4" ) ; final VersionNumber scala2_11_0 = new VersionNumber ( "2 . 11 . 0" ) ; final VersionNumber scala2_9_0 = new VersionNumber ( "2 . 9 . 0" ) ; if ( scalaVersion . major == 3 ) { return super . factory . createArtifact ( "org . jline" , ScalaConsoleMojo . JLINE , "3 . 19 . 0" , "" , MavenArtifactResolver . JAR ) ; } else if ( scala2_12_0M4 . compareTo ( scalaVersion ) < = 0 ) { return super . factory . createArtifact ( ScalaConsoleMojo . JLINE , ScalaConsoleMojo . JLINE , "2 . 14 . 1" , "" , MavenArtifactResolver . JAR ) ; } else if ( scala2_11_0 . compareTo ( scalaVersion ) < = 0 ) { return super . factory . createArtifact ( ScalaConsoleMojo . JLINE , ScalaConsoleMojo . JLINE , "2 . 12" , "" , MavenArtifactResolver . JAR ) ; } else if ( scala2_9_0 . compareTo ( scalaVersion ) < = 0 ) { return super . factory . createArtifact ( ScalaConsoleMojo . SCALA_ORG_GROUP , ScalaConsoleMojo . JLINE , scalaVersion . toString ( ) , "" , MavenArtifactResolver . JAR ) ; } else { return super . factory . createArtifact ( ScalaConsoleMojo . JLINE , ScalaConsoleMojo . JLINE , "0 . 9 . 94" , "" , MavenArtifactResolver . JAR ) ; } } }