#ifndef __MEM_H__
#define __MEM_H__

// http://stackoverflow.com/questions/6352206/aligned-calloc-visual-studio
void* _mm_calloc(size_t nelem, size_t elsize, size_t alignment)
{
    // Watch out for overflow
    if(elsize == 0)
        return NULL;

    size_t size = nelem * elsize;
    void* memory = _mm_malloc(size, alignment);
    if(memory != NULL)
        memset(memory, 0, size);
    return memory;
}

#endif // __MEM_H__
