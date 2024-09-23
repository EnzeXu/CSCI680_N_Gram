public class SunLogger extends AbstractLogger { private final java.util.logging.Logger sunLogger; public SunLogger(String name) { super(name); sunLogger = java.util.logging.Logger.getLogger(name); } @Override public boolean isTraceEnabled() { return (sunLogger.isLoggable(java.util.logging.Level.FINEST)); } @Override public boolean isDebugEnabled() { return (sunLogger.isLoggable(java.util.logging.Level.FINE)); } @Override public boolean isInfoEnabled() { return (sunLogger.isLoggable(java.util.logging.Level.INFO)); } @Override public void log(Level level, Object message, Throwable e) { java.util.logging.Level sLevel = java.util.logging.Level.SEVERE; switch (level == null ? Level.FATAL : level) { case TRACE: sLevel = java.util.logging.Level.FINEST; break; case DEBUG: sLevel = java.util.logging.Level.FINE; break; case INFO: sLevel = java.util.logging.Level.INFO; break; case WARN: sLevel = java.util.logging.Level.WARNING; break; case ERROR: sLevel = java.util.logging.Level.SEVERE; break; case FATAL: sLevel = java.util.logging.Level.SEVERE; break; default: sLevel = java.util.logging.Level.SEVERE; sunLogger.log(sLevel, "Unhandled log level: " + level + " for the following message"); } Throwable t = new Throwable(); StackTraceElement[] ste = t.getStackTrace(); StackTraceElement logRequestor = null; String alclass = AbstractLogger.class.getName(); for (int i = 0; i < ste.length && logRequestor == null; i++) { if (ste[i].getClassName().equals(alclass)) { if (i + 1 < ste.length) { logRequestor = ste[i + 1]; if (logRequestor.getClassName().equals(alclass)) { logRequestor = null; } } } } if (logRequestor != null) { if (e != null) { sunLogger.logp(sLevel, logRequestor.getClassName(), logRequestor.getMethodName(), message.toString(), e); } else { sunLogger.logp(sLevel, logRequestor.getClassName(), logRequestor.getMethodName(), message.toString()); } } else { if (e != null) { sunLogger.log(sLevel, message.toString(), e); } else { sunLogger.log(sLevel, message.toString()); } } } }