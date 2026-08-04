#pragma once

/** @file
 * Small Objective-C ownership helpers shared by Apple Objective-C++ backends.
 */

#ifdef __OBJC__

#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>

#include <cstddef>
#include <cstdint>
#include <unistd.h>

namespace Darknet::Apple::Detail
{
	/** Metal may alias host storage only when both the address and length are
	 * aligned to the host VM page size. Keep this check shared by inference and
	 * training so invalid pointers never reach newBufferWithBytesNoCopy.
	 */
	inline bool no_copy_buffer_eligible(const void *ptr, const size_t bytes)
	{
		const long page_size = sysconf(_SC_PAGESIZE);
		return ptr && bytes > 0 && page_size > 0 &&
			(reinterpret_cast<uintptr_t>(ptr) % static_cast<size_t>(page_size)) == 0 &&
			(bytes % static_cast<size_t>(page_size)) == 0;
	}

	/** Strong Objective-C reference suitable for use in C++ containers. */
	class ObjcStrong
	{
	public:
		ObjcStrong() = default;

		explicit ObjcStrong(id value) : object_((__bridge CFTypeRef)value)
		{
			retain();
		}

		ObjcStrong(const ObjcStrong & other) : object_(other.object_)
		{
			retain();
		}

		ObjcStrong(ObjcStrong && other) noexcept : object_(other.object_)
		{
			other.object_ = nullptr;
		}

		ObjcStrong & operator=(const ObjcStrong & other)
		{
			if (this != &other)
			{
				release();
				object_ = other.object_;
				retain();
			}
			return *this;
		}

		ObjcStrong & operator=(ObjcStrong && other) noexcept
		{
			if (this != &other)
			{
				release();
				object_ = other.object_;
				other.object_ = nullptr;
			}
			return *this;
		}

		~ObjcStrong()
		{
			release();
		}

		id get() const
		{
			return (__bridge id)object_;
		}

	private:
		void retain()
		{
			if (object_)
			{
				CFRetain(object_);
			}
		}

		void release()
		{
			if (object_)
			{
				CFRelease(object_);
				object_ = nullptr;
			}
		}

		CFTypeRef object_ = nullptr;
	};

	inline void objc_release_if_needed(id object)
	{
#if !__has_feature(objc_arc)
		[object release];
#else
		(void)object;
#endif
	}
}

#endif
