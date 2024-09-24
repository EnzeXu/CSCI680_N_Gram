public class PlatformInfo implements Serializable, Comparable<PlatformInfo> { public static final PlatformInfo NULL = new PlatformInfo( null, null, null ); public final String name; public final String vendor; public final String version; public PlatformInfo( String name, String vendor, String version ) { this.name = name; this.vendor = vendor; this.version = version; } @Override public int compareTo( PlatformInfo other ) { if( other == null ) return 1; return this.toString().compareTo( other.toString() ); } @Override public boolean equals( Object object ) { if( this == object ) return true; if( object == null || getClass() != object.getClass() ) return false; PlatformInfo that = (PlatformInfo) object; if( name != null ? !name.equals( that.name ) : that.name != null ) return false; if( vendor != null ? !vendor.equals( that.vendor ) : that.vendor != null ) return false; if( version != null ? !version.equals( that.version ) : that.version != null ) return false; return true; } @Override public int hashCode() { int result = name != null ? name.hashCode() : 0; result = 31 * result + ( vendor != null ? vendor.hashCode() : 0 ); result = 31 * result + ( version != null ? version.hashCode() : 0 ); return result; } @Override public String toString() { final StringBuilder sb = new StringBuilder(); if( name == null ) sb.append( "UNKNOWN" ).append( ':' ); else sb.append( name ).append( ':' ); if( version != null ) sb.append( version ).append( ':' ); if( vendor != null ) sb.append( vendor ); return sb.toString(); } }