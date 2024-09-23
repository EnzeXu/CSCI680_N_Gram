public class GlobHfs extends MultiSourceTap<Hfs, Configuration, RecordReader> { private final String pathPattern; private final PathFilter pathFilter; @ConstructorProperties({"scheme", "pathPattern"}) public GlobHfs( Scheme<Configuration, RecordReader, ?, ?, ?> scheme, String pathPattern ) { this( scheme, pathPattern, null ); } @ConstructorProperties({"scheme", "pathPattern", "pathFilter"}) public GlobHfs( Scheme<Configuration, RecordReader, ?, ?, ?> scheme, String pathPattern, PathFilter pathFilter ) { super( scheme ); this.pathPattern = pathPattern; this.pathFilter = pathFilter; } @Override public String getIdentifier() { return pathPattern; } @Override protected Hfs[] getTaps() { return initTapsInternal( new JobConf() ); } private Hfs[] initTapsInternal( Configuration conf ) { if( taps != null ) return taps; try { taps = makeTaps( conf ); } catch( IOException exception ) { throw new TapException( "unable to resolve taps for globing path: " + pathPattern ); } return taps; } private Hfs[] makeTaps( Configuration conf ) throws IOException { FileStatus[] statusList; Path path = new Path( pathPattern ); FileSystem fileSystem = path.getFileSystem( conf ); if( pathFilter == null ) statusList = fileSystem.globStatus( path ); else statusList = fileSystem.globStatus( path, pathFilter ); if( statusList == null || statusList.length == 0 ) throw new TapException( "unable to find paths matching path pattern: " + pathPattern ); List<Hfs> notEmpty = new ArrayList<Hfs>(); for( int i = 0; i < statusList.length; i++ ) { if( statusList[ i ].isDir() || statusList[ i ].getLen() != 0 ) notEmpty.add( new Hfs( getScheme(), statusList[ i ].getPath().toString() ) ); } if( notEmpty.isEmpty() ) throw new TapException( "all paths matching path pattern are zero length and not directories: " + pathPattern ); return notEmpty.toArray( new Hfs[ notEmpty.size() ] ); } @Override public void sourceConfInit( FlowProcess<? extends Configuration> process, Configuration conf ) { Hfs[] taps = initTapsInternal( conf ); taps[ 0 ].sourceConfInitAddInputPaths( conf, new LazyIterable<Hfs, Path>( taps ) { @Override protected Path convert( Hfs next ) { return next.getPath(); } } ); taps[ 0 ].sourceConfInitComplete( process, conf ); } @Override public boolean equals( Object object ) { if( this == object ) return true; if( object == null || getClass() != object.getClass() ) return false; GlobHfs globHfs = (GlobHfs) object; if( getScheme() != null ? !getScheme().equals( globHfs.getScheme() ) : globHfs.getScheme() != null ) return false; if( pathFilter != null ? !pathFilter.equals( globHfs.pathFilter ) : globHfs.pathFilter != null ) return false; if( pathPattern != null ? !pathPattern.equals( globHfs.pathPattern ) : globHfs.pathPattern != null ) return false; return true; } @Override public int hashCode() { int result = pathPattern != null ? pathPattern.hashCode() : 0; result = 31 * result + ( pathFilter != null ? pathFilter.hashCode() : 0 ); return result; } @Override public String toString() { return "GlobHfs[" + pathPattern + ']'; } }