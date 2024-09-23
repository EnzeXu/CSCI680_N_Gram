public class UtilHadoopTest { @Test public void testSetLogLevel() { Configuration jobConf = new Configuration(); jobConf.set( "log4j.logger", "cascading=DEBUG" ); HadoopUtil.initLog4j( jobConf ); Object loggerObject = Util.invokeStaticMethod( "org.apache.log4j.Logger", "getLogger", new Object[]{"cascading"}, new Class[]{String.class} ); Object levelObject = Util.invokeStaticMethod( "org.apache.log4j.Level", "toLevel", new Object[]{"DEBUG"}, new Class[]{String.class} ); Object returnedLevel = Util.invokeInstanceMethod( loggerObject, "getLevel", new Object[]{}, new Class[]{} ); assertEquals( levelObject, returnedLevel ); } }