public class StdInTap extends SourceTap < Properties , InputStream > { public StdInTap ( Scheme < Properties , InputStream , ? , ? , ? > scheme ) { super ( scheme ) ; } @ Override public String getIdentifier ( ) { return "stdIn" ; } @ Override public TupleEntryIterator openForRead ( FlowProcess < ? extends Properties > flowProcess , InputStream inputStream ) throws IOException { return new TupleEntrySchemeIterator < Properties , InputStream > ( flowProcess , this , getScheme ( ) , System . in ) ; } @ Override public boolean resourceExists ( Properties conf ) throws IOException { return true ; } @ Override public long getModifiedTime ( Properties conf ) throws IOException { return System . currentTimeMillis ( ) ; } }