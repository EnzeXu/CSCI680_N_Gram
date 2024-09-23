public class Trie<V extends Serializable> implements Serializable { static public class Entry<V> implements Serializable { String prefix; V value; public Entry() { } public Entry( String p, V v ) { prefix = p; value = v; } public String prefix() { return prefix; } public V value() { return value; } @Override public String toString() { final StringBuilder sb = new StringBuilder( "Entry{" ); sb.append( "prefix='" ).append( prefix ).append( '\'' ); sb.append( ", value=" ).append( value ); sb.append( '}' ); return sb.toString(); } } Entry<V> entry; char key; Map<Character, Trie<V>> children; public Trie() { this.children = new HashMap<>(); this.entry = new Entry<>(); } Trie( char key ) { this.children = new HashMap<>(); this.key = key; entry = new Entry<V>(); } public void put( String key, V value ) { put( new StringBuffer( key ), new StringBuffer( "" ), value ); } void put( StringBuffer remainder, StringBuffer prefix, V value ) { if( remainder.length() <= 0 ) { this.entry.value = value; this.entry.prefix = prefix.toString(); return; } char keyElement = remainder.charAt( 0 ); Trie<V> trie = null; try { trie = children.get( keyElement ); } catch( IndexOutOfBoundsException ignored ) { } if( trie == null ) { trie = new Trie<>( keyElement ); children.put( keyElement, trie ); } prefix.append( remainder.charAt( 0 ) ); trie.put( remainder.deleteCharAt( 0 ), prefix, value ); } public V get( String key ) { return get( new StringBuffer( key ), 0 ); } public boolean hasPrefix( String key ) { return ( this.get( key ) != null ); } V get( StringBuffer key, int level ) { if( key.length() <= 0 ) return entry.value; Trie<V> trie = children.get( key.charAt( 0 ) ); if( trie != null ) return trie.get( key.deleteCharAt( 0 ), ++level ); else return ( level > 0 ) ? entry.value : null; } public String getCommonPrefix() { StringBuffer buffer = new StringBuffer(); if( children.size() != 1 ) return buffer.toString(); buildPrefix( buffer, this ); return buffer.toString(); } private void buildPrefix( StringBuffer buffer, Trie<V> current ) { if( current.children.size() != 1 ) return; for( Map.Entry<Character, Trie<V>> entry : current.children.entrySet() ) { buffer.append( entry.getKey() ); buildPrefix( buffer, entry.getValue() ); } } @Override public String toString() { final StringBuilder sb = new StringBuilder( "Trie{" ); sb.append( "entry=" ).append( entry ); sb.append( ", key=" ).append( key ); sb.append( ", children=" ).append( children ); sb.append( '}' ); return sb.toString(); } }