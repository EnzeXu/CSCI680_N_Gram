public class StdErrTap extends SinkTap < Properties , OutputStream > { public StdErrTap ( Scheme < Properties , ? , OutputStream , ? , ? > scheme ) { super ( scheme , SinkMode . UPDATE ) ; } @ Override public String getIdentifier ( ) { return "stdErr" ; } @ Override public TupleEntryCollector openForWrite ( FlowProcess < ? extends Properties > flowProcess , OutputStream output ) throws IOException { return new TupleEntrySchemeCollector < Properties , OutputStream > ( flowProcess , this , getScheme ( ) , System . err ) ; } @ Override public boolean createResource ( Properties conf ) throws IOException { return true ; } @ Override public boolean deleteResource ( Properties conf ) throws IOException { return false ; } @ Override public boolean resourceExists ( Properties conf ) throws IOException { return true ; } @ Override public long getModifiedTime ( Properties conf ) throws IOException { return 0 ; } }