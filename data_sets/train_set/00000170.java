public class ScalaHelpMojo extends ScalaMojoSupport { @Parameter(property = "maven.scala.help.versionOnly", defaultValue = "false") private boolean versionOnly; protected JavaMainCaller getScalaCommand() throws Exception { Context sc = findScalaContext(); return getScalaCommand(fork, sc.compilerMainClassName(scalaClassName, false)); } @Override public void doExecute() throws Exception { JavaMainCaller jcmd; if (!versionOnly) { jcmd = getScalaCommand(); jcmd.addArgs("-help"); jcmd.addArgs("-X"); jcmd.addArgs("-Y"); jcmd.run(displayCmd); } jcmd = getScalaCommand(); jcmd.addArgs("-version"); jcmd.run(displayCmd); } }