public class LocalConfigDefScheme extends cascading.scheme.local.TextLine { public LocalConfigDefScheme( Fields sourceFields ) { super( sourceFields ); } @Override public void sourceConfInit( FlowProcess<? extends Properties> flowProcess, Tap<Properties, InputStream, OutputStream> tap, Properties conf ) { if( flowProcess.getProperty( "default" ) != null ) throw new RuntimeException( "default should be null" ); super.sourceConfInit( flowProcess, tap, conf ); } @Override public void sinkConfInit( FlowProcess<? extends Properties> flowProcess, Tap<Properties, InputStream, OutputStream> tap, Properties conf ) { if( flowProcess.getProperty( "default" ) != null ) throw new RuntimeException( "default should be null" ); super.sinkConfInit( flowProcess, tap, conf ); } @Override public void sourcePrepare( FlowProcess<? extends Properties> flowProcess, SourceCall<LineNumberReader, InputStream> sourceCall ) throws IOException { if( !( flowProcess instanceof FlowProcessWrapper ) ) throw new RuntimeException( "not a flow process wrapper" ); if( !"process-default".equals( flowProcess.getProperty( "default" ) ) ) throw new RuntimeException( "not default value" ); if( !"source-replace".equals( flowProcess.getProperty( "replace" ) ) ) throw new RuntimeException( "not replaced value" ); flowProcess = ( (FlowProcessWrapper) flowProcess ).getDelegate(); if( !"process-default".equals( flowProcess.getProperty( "default" ) ) ) throw new RuntimeException( "not default value" ); if( !"process-replace".equals( flowProcess.getProperty( "replace" ) ) ) throw new RuntimeException( "not replaced value" ); super.sourcePrepare( flowProcess, sourceCall ); } @Override public void sinkPrepare( FlowProcess<? extends Properties> flowProcess, SinkCall<PrintWriter, OutputStream> sinkCall ) throws IOException { if( !( flowProcess instanceof FlowProcessWrapper ) ) throw new RuntimeException( "not a flow process wrapper" ); if( !"process-default".equals( flowProcess.getProperty( "default" ) ) ) throw new RuntimeException( "not default value" ); if( !"sink-replace".equals( flowProcess.getProperty( "replace" ) ) ) throw new RuntimeException( "not replaced value" ); flowProcess = ( (FlowProcessWrapper) flowProcess ).getDelegate(); if( !"process-default".equals( flowProcess.getProperty( "default" ) ) ) throw new RuntimeException( "not default value" ); if( !"process-replace".equals( flowProcess.getProperty( "replace" ) ) ) throw new RuntimeException( "not replaced value" ); super.sinkPrepare( flowProcess, sinkCall ); } }