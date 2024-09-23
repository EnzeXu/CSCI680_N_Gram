public class BaseMapReduceFlow extends HadoopFlow { protected boolean deleteSinkOnInit = false ; protected BaseMapReduceFlow ( PlatformInfo platformInfo , Map < Object , Object > properties , JobConf jobConf , String name , Map < String , String > flowDescriptor , boolean deleteSinkOnInit ) { super ( platformInfo , properties , jobConf , name , flowDescriptor ) ; this . deleteSinkOnInit = deleteSinkOnInit ; } protected BaseMapReduceFlow ( PlatformInfo platformInfo , Map < Object , Object > properties , String name , Map < String , String > flowDescriptor , boolean deleteSinkOnInit ) { super ( platformInfo , properties , new JobConf ( ) , name , flowDescriptor ) ; this . deleteSinkOnInit = deleteSinkOnInit ; } protected FlowStepGraph makeStepGraph ( JobConf jobConf ) { FlowStepGraph flowStepGraph = new FlowStepGraph ( ) ; Tap sink = getSinksCollection ( ) . iterator ( ) . next ( ) ; FlowStep < JobConf > step = createFlowStep ( jobConf , sink ) ; flowStepGraph . addVertex ( step ) ; return flowStepGraph ; } protected FlowStep < JobConf > createFlowStep ( JobConf jobConf , Tap sink ) { return new MapReduceFlowStep ( this , sink . toString ( ) , jobConf , sink ) ; } protected Map < String , Tap > createSources ( JobConf jobConf ) { return fileInputToTaps ( jobConf ) ; } protected Map < String , Tap > fileInputToTaps ( JobConf jobConf ) { Path [ ] paths = FileInputFormat . getInputPaths ( jobConf ) ; if ( paths == null || paths . length == 0 ) { try { paths = org . apache . hadoop . mapreduce . lib . input . FileInputFormat . getInputPaths ( new Job ( jobConf ) ) ; } catch ( IOException exception ) { throw new CascadingException ( exception ) ; } } Map < String , Tap > taps = new HashMap < > ( ) ; if ( paths == null ) return taps ; for ( Path path : paths ) toSourceTap ( jobConf , taps , path ) ; return taps ; } protected Tap toSourceTap ( JobConf jobConf , Map < String , Tap > taps , Path path ) { String name = makeNameFromPath ( taps , path ) ; return taps . put ( name , createTap ( jobConf , path , SinkMode . KEEP ) ) ; } protected Map < String , Tap > createSinks ( JobConf jobConf ) { return fileOutputToTaps ( jobConf ) ; } protected Map < String , Tap > fileOutputToTaps ( JobConf jobConf ) { Path path = FileOutputFormat . getOutputPath ( jobConf ) ; if ( path == null ) { try { path = org . apache . hadoop . mapreduce . lib . output . FileOutputFormat . getOutputPath ( new Job ( jobConf ) ) ; } catch ( IOException exception ) { throw new CascadingException ( exception ) ; } } Map < String , Tap > taps = new HashMap < > ( ) ; if ( path != null ) toSinkTap ( jobConf , taps , path ) ; return taps ; } protected Tap toSinkTap ( JobConf jobConf , Map < String , Tap > taps , Path path ) { String name = makeNameFromPath ( taps , path ) ; SinkMode sinkMode = deleteSinkOnInit ? SinkMode . REPLACE : SinkMode . KEEP ; return taps . put ( name , createTap ( jobConf , path , sinkMode ) ) ; } protected Tap createTap ( JobConf jobConf , Path path , SinkMode sinkMode ) { return new Hfs ( new NullScheme ( ) , path . toString ( ) , sinkMode ) ; } protected String makeNameFromPath ( Map < String , Tap > taps , Path path ) { Path parent = path . getParent ( ) ; String name = path . getName ( ) ; while ( taps . containsKey ( name ) ) { name = new Path ( parent . getName ( ) , name ) . toString ( ) ; parent = parent . getParent ( ) ; } return name ; } protected Map < String , Tap > createTraps ( JobConf jobConf ) { return new HashMap < > ( ) ; } }