// Copyright (c) 2024 COMPAILE Solutions GmbH - Grischa Hauser
// License:MIT License See LICENSE for the full license.
// https://github.com/Compaile/ctrack
#pragma once
#ifndef CTRACK_H
#define CTRACK_H

#include <string>
#include <iostream>
#include <iterator>
#include <thread>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <cstdint>
#include <deque>
#include <map>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <numeric>
#ifndef CTRACK_DISABLE_EXECUTION_POLICY
#include <execution>
#endif
#include <vector>
#include <iomanip>
#include <filesystem>
#include <sstream>
#include <atomic>
#include <cmath>

#define CTRACK_VERSION_MAJOR 1
#define CTRACK_VERSION_MINOR 1
#define CTRACK_VERSION_PATCH 0

// Helper macro to convert a numeric value to a string
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// Create a string version
#define CTRACK_VERSION_STRING      \
	TOSTRING(CTRACK_VERSION_MAJOR) \
	"_" TOSTRING(CTRACK_VERSION_MINOR) "_" TOSTRING(CTRACK_VERSION_PATCH)

// Use the version string as the namespace name
#define CTRACK_VERSION_NAMESPACE v##CTRACK_VERSION_MAJOR##_##CTRACK_VERSION_MINOR##_##CTRACK_VERSION_PATCH

namespace ctrack
{

	inline namespace CTRACK_VERSION_NAMESPACE
	{
#ifndef CTRACK_DISABLE_EXECUTION_POLICY
		constexpr auto execution_policy = std::execution::par_unseq;
#define OPT_EXEC_POLICY execution_policy,
#else
#define OPT_EXEC_POLICY
#endif

		template <typename T, typename Field>
		auto sum_field(const std::vector<T> &vec, Field T::*field)
		{
			using FieldType = std::decay_t<decltype(std::declval<T>().*field)>;
			return std::transform_reduce(
				OPT_EXEC_POLICY
					vec.begin(),
				vec.end(),
				FieldType{},
				std::plus<>(),
				[field](const auto &item)
				{ return item.*field; });
		}

		template <typename T, typename Field>
		auto sum_squared_field(const std::vector<T> &values, Field T::*field)
		{
			using FieldType = std::decay_t<decltype(std::declval<T>().*field)>;
			return std::transform_reduce(
				OPT_EXEC_POLICY
					values.begin(),
				values.end(),
				FieldType{},
				std::plus<>(),
				[field](const T &v)
				{
					return (v.*field) * (v.*field);
				});
		}

		template <typename T, typename Field>
		double calculate_std_dev_field(std::vector<T> &values, Field T::*field, const double mean)
		{
			double res = std::transform_reduce(
				OPT_EXEC_POLICY
					values.begin(),
				values.end(),
				0.0,
				std::plus<>(),
				[mean, field](const T &v)
				{
					return std::pow(static_cast<double>(v.*field) - mean, 2);
				});

			return sqrt(res / values.size());
		}

		template <typename T, typename Field>
		auto get_distinct_field_values(const std::vector<const T *> &vec, Field T::*field)
		{
			std::set<std::remove_reference_t<decltype(std::declval<T>().*field)>> distinct_values;

			std::transform(vec.begin(), vec.end(),
						   std::inserter(distinct_values, distinct_values.end()),
						   [field](const T *item)
						   { return item->*field; });
			return distinct_values;
		}

		template <typename T, typename Field>
		auto get_distinct_field_values(const std::vector<T *> &vec, Field T::*field)
		{
			std::set<std::remove_reference_t<decltype(std::declval<T>().*field)>> distinct_values;

			std::transform(vec.begin(), vec.end(),
						   std::inserter(distinct_values, distinct_values.end()),
						   [field](const T *item)
						   { return item->*field; });
			return distinct_values;
		}

		template <typename T, typename Field>
		auto get_distinct_field_values(const std::vector<T> &vec, Field T::*field)
		{
			std::set<std::remove_reference_t<decltype(std::declval<T>().*field)>> distinct_values;

			std::transform(vec.begin(), vec.end(),
						   std::inserter(distinct_values, distinct_values.end()),
						   [field](const T &item)
						   { return item.*field; });
			return distinct_values;
		}

		template <typename T, typename Field>
		size_t count_distinct_field_values(const std::vector<T> &vec, Field T::*field)
		{
			std::unordered_set<std::remove_reference_t<decltype(std::declval<T>().*field)>> distinct_values;
			distinct_values.reserve(vec.size() / 10); // Heuristic pre-allocation
			for (const auto *item : vec)
			{
				distinct_values.insert(item->*field);
			}
			return distinct_values.size();
		}

		template <typename StructType, typename MemberType>
		void order_pointer_vector_by_field(std::vector<StructType *> &vec, MemberType StructType::*member, bool asc = true)
		{
			std::sort(OPT_EXEC_POLICY vec.begin(), vec.end(),
					  [member, asc](const StructType *a, const StructType *b)
					  {
						  if (asc)
							  return (a->*member) < (b->*member);
						  else
							  return (a->*member) > (b->*member);
					  });
		}

		template <typename T>
		size_t countAllEvents(const std::deque<std::vector<T>> &events)
		{
			return std::transform_reduce(
				OPT_EXEC_POLICY
					events.begin(),
				events.end(),
				size_t(0),
				std::plus<>(),
				[](const auto &vec)
				{
					return vec.size();
				});
		}

		struct ColorScheme
		{
			std::string border_color;
			std::string header_color;
			std::string top_header_color;
			std::string row_color;

			ColorScheme(const std::string &border,
						const std::string &header,
						const std::string &top_header,
						const std::string &row)
				: border_color(border),
				  header_color(header),
				  top_header_color(top_header),
				  row_color(row) {}
		};

		static inline const ColorScheme default_colors{
			"\033[38;5;24m",	// Darker Blue (Border)
			"\033[1;38;5;135m", // Purple (Header)
			"\033[1;38;5;92m",	// Darker Purple (Top Header)
			"\033[38;5;39m"		// Light Blue (Row)
		};

		// Alternate color scheme (still nice to read on terminals)
		static inline const ColorScheme alternate_colors{
			"\033[38;5;28m",	// Dark Green (Border)
			"\033[1;38;5;208m", // Orange (Header)
			"\033[1;38;5;130m", // Dark Orange (Top Header)
			"\033[38;5;71m"		// Light Green (Row)
		};

		class BeautifulTable
		{
		private:
			std::vector<std::pair<std::string, int>> top_header;
			std::vector<std::string> header;
			std::vector<std::vector<std::string>> rows;
			std::vector<size_t> columnWidths;
			bool useColor;
			ColorScheme colors;
			static inline const std::string RESET_COLOR = "\033[0m";

			void updateColumnWidths(const std::vector<std::string> &row)
			{
				for (size_t i = 0; i < row.size(); ++i)
				{
					if (i >= columnWidths.size())
					{
						columnWidths.push_back(row[i].length());
					}
					else
					{
						columnWidths[i] = std::max<size_t>(columnWidths[i], row[i].length());
					}
				}
			}

			template <typename StreamType>
			void printHorizontalLine(StreamType &stream) const
			{
				if (useColor)
					stream << colors.border_color;
				stream << "+";
				for (size_t width : columnWidths)
				{
					stream << std::string(width + 2, '-') << "+";
				}
				if (useColor)
					stream << RESET_COLOR;
				stream << "\n";
			}

			template <typename StreamType>
			void printRow(StreamType &stream, const std::vector<std::string> &row, const std::string &color, bool center = false) const
			{
				if (useColor)
					stream << colors.border_color;
				stream << "|";
				if (useColor)
					stream << RESET_COLOR << color;
				for (size_t i = 0; i < row.size(); ++i)
				{
					if (center)
					{
						size_t padding = columnWidths[i] - row[i].length();
						size_t leftPadding = padding / 2;
						size_t rightPadding = padding - leftPadding;
						stream << std::string(leftPadding + 1, ' ') << row[i] << std::string(rightPadding + 1, ' ');
					}
					else
					{
						stream << " " << std::setw(static_cast<int32_t>(columnWidths[i])) << std::right << row[i] << " ";
					}
					if (useColor)
						stream << RESET_COLOR << colors.border_color;
					stream << "|";
					if (useColor)
						stream << RESET_COLOR << color;
				}
				if (useColor)
					stream << RESET_COLOR;
				stream << "\n";
			}

			template <typename StreamType>
			void printRow(StreamType &stream, const std::vector<std::pair<std::string, int>> &row, const std::string &color) const
			{
				if (useColor)
					stream << colors.border_color;
				stream << "|";
				if (useColor)
					stream << RESET_COLOR << color;
				int y = 0;
				for (size_t i = 0; i < row.size(); ++i)
				{
					size_t sum = row[i].second - 1;
					for (int x = y; x < y + row[i].second; x++)
					{
						sum += columnWidths[x] + 2;
					}
					y += row[i].second;

					size_t textWidth = row[i].first.length();
					size_t totalPadding = sum - textWidth;
					size_t leftPadding = totalPadding / 2;
					size_t rightPadding = totalPadding - leftPadding;

					// Print left padding
					stream << std::string(leftPadding, ' ');

					// Print text
					stream << row[i].first;

					// Print right padding
					stream << std::string(rightPadding, ' ');
					if (useColor)
						stream << RESET_COLOR << colors.border_color;
					stream << "|";
					if (useColor)
						stream << RESET_COLOR << color;
				}
				if (useColor)
					stream << RESET_COLOR;
				stream << "\n";
			}

		public:
			BeautifulTable(const std::vector<std::string> &headerColumns, bool enableColor = false, const ColorScheme &colors = default_colors, const std::vector<std::pair<std::string, int>> &top_header = {})
				: top_header(top_header), header(headerColumns), useColor(enableColor), colors(colors)
			{
				updateColumnWidths(header);
			}

			void addRow(const std::vector<std::string> &row)
			{
				if (row.size() != header.size())
				{
					throw std::invalid_argument("Row size must match header size");
				}
				rows.push_back(row);
				updateColumnWidths(row);
			}

			template <typename StreamType>
			void print(StreamType &stream) const
			{
				if (top_header.size() > 0)
				{
					printHorizontalLine(stream);
					printRow(stream, top_header, colors.top_header_color);
				}
				printHorizontalLine(stream);
				printRow(stream, header, colors.header_color, true);
				printHorizontalLine(stream);
				for (const auto &row : rows)
				{
					printRow(stream, row, colors.row_color);
					printHorizontalLine(stream);
				}
			}

			template <typename T>
			static inline std::string table_string(const T &value)
			{
				std::ostringstream oss;
				oss << value;
				return oss.str();
			}

			static inline std::string table_time(uint_fast64_t nanoseconds)
			{
				return table_time(static_cast<double>(nanoseconds));
			}

			static inline std::string table_time(double nanoseconds)
			{
				const char *units[] = {"ns", "mcs", "ms", "s"};
				int unit = 0;
				double value = static_cast<double>(nanoseconds);
				while (value >= 1000 && unit < 3)
				{
					value /= 1000;
					unit++;
				}
				std::ostringstream oss;
				oss << std::fixed << std::setprecision(2) << value << " " << units[unit];
				return oss.str();
			}

			static inline std::string table_percentage(uint_fast64_t value, uint_fast64_t total)
			{
				if (total == 0)
				{
					return "nan%";
				}

				// Calculate the percentage
				double percentage = (static_cast<double>(value) / total) * 100.0;

				// Format the percentage as a string with 2 decimal places
				std::ostringstream ss;
				ss << std::fixed << std::setprecision(2) << percentage << "%";

				return ss.str();
			}

			static inline std::string table_timepoint(const std::chrono::high_resolution_clock::time_point &tp)
			{
				auto system_tp = std::chrono::system_clock::now() +
								 std::chrono::duration_cast<std::chrono::system_clock::duration>(
									 tp - std::chrono::high_resolution_clock::now());

				auto tt = std::chrono::system_clock::to_time_t(system_tp);
				std::tm tm{};

#if defined(_WIN32)
				localtime_s(&tm, &tt);
#else
				localtime_r(&tt, &tm);
#endif

				std::ostringstream oss;
				oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
				return oss.str();
			}

			static inline std::string stable_shortenPath(const std::string &fullPath, size_t maxLength = 35)
			{
				namespace fs = std::filesystem;

				fs::path path(fullPath);
				std::string filename = path.filename().string();

				if (filename.length() <= maxLength)
				{
					return filename;
				}

				// If filename is too long, truncate it and add ...
				return filename.substr(0, maxLength - 3) + "...";
			}

			using bt = BeautifulTable;
		};

		struct Event
		{
			std::chrono::high_resolution_clock::time_point start_time;
			std::chrono::high_resolution_clock::time_point end_time;
			int line;
			int thread_id;
			std::string_view filename;
			std::string_view function;
			unsigned int event_id; // This ID is 1-based and unique per thread
			Event(const std::chrono::high_resolution_clock::time_point &start_time, const std::chrono::high_resolution_clock::time_point &end_time, const std::string_view filename, const int line, const std::string_view function, const int thread_id, const unsigned int event_id)
				: start_time(start_time), end_time(end_time), line(line), thread_id(thread_id), filename(filename), function(function), event_id(event_id)
			{
			}
		};

		struct Simple_Event
		{
			uint_fast64_t duration = 0;
			std::chrono::high_resolution_clock::time_point start_time{};
			int_fast64_t unique_id = 0;
			std::chrono::high_resolution_clock::time_point end_time{};
			Simple_Event(const std::chrono::high_resolution_clock::time_point &start_time, const std::chrono::high_resolution_clock::time_point &end_time, const uint_fast64_t duration, const int_fast64_t unique_id) : duration(duration), start_time(start_time), unique_id(unique_id), end_time(end_time) {}
			Simple_Event() {}
		};

		inline bool cmp_simple_event_by_duration_asc(const Simple_Event &a, const Simple_Event &b)
		{
			return a.duration < b.duration;
		}
		inline bool cmp_simple_event_by_start_time_asc(const Simple_Event &a, const Simple_Event &b)
		{
			return a.start_time < b.start_time;
		}

		inline uint_fast64_t get_unique_event_id(unsigned int thread_id, unsigned int event_id)
		{
			uint_fast64_t uniqueId = static_cast<uint_fast64_t>(thread_id);
			uniqueId = uniqueId << 32;
			uniqueId += static_cast<uint_fast64_t>(event_id);
			return uniqueId;
		}

		// Helper to unpack the combined event ID back into its components.
		inline std::pair<unsigned int, unsigned int> unpack_unique_event_id(int_fast64_t unique_id)
		{
			unsigned int thread_id = static_cast<unsigned int>(unique_id >> 32);
			unsigned int event_id = static_cast<unsigned int>(unique_id & 0xFFFFFFFF);
			return {thread_id, event_id};
		}

		inline std::vector<Simple_Event> create_simple_events(const std::vector<const Event *> &events)
		{
			std::vector<Simple_Event> simple_events{};
			simple_events.resize(events.size());
			std::transform(
				OPT_EXEC_POLICY
					events.begin(),
				events.end(),
				simple_events.begin(),
				[](const Event *event)
				{
					Simple_Event simple_event(event->start_time, event->end_time, std::chrono::duration_cast<std::chrono::nanoseconds>(event->end_time - event->start_time).count(), get_unique_event_id(event->thread_id, event->event_id));
					return simple_event;
				});
			return simple_events;
		}

		// requires already sorted
		inline std::vector<Simple_Event> sorted_create_grouped_simple_events(const std::vector<Simple_Event> &events)
		{
			std::vector<Simple_Event> result{};
			if (events.size() == 0)
				return result;
			result.push_back(events[0]);
			unsigned int current_idx = 0;

			for (size_t i = 1; i < events.size(); i++)
			{
				if (result[current_idx].end_time >= events[i].start_time)
				{
					result[current_idx].end_time = std::max<std::chrono::high_resolution_clock::time_point>(result[current_idx].end_time, events[i].end_time);
				}
				else
				{
					result.push_back(events[i]);
					current_idx++;
				}
			}

			for (auto &entry : result)
			{
				entry.duration = std::chrono::duration_cast<std::chrono::nanoseconds>(entry.end_time - entry.start_time).count();
			}

			return result;
		}

		typedef std::vector<Event> t_events;

		inline std::vector<Simple_Event> load_child_events_simple(const std::vector<Simple_Event> &parent_events_simple,
																  const std::deque<t_events> &all_events_storage,
																  const std::unordered_map<int_fast64_t, std::vector<int_fast64_t>> &child_graph)
		{
			std::vector<const Event *> child_events{};
			child_events.reserve(parent_events_simple.size()); // Pre-allocate a reasonable guess

			for (const auto &simple_parent_event : parent_events_simple)
			{
				auto it = child_graph.find(simple_parent_event.unique_id);
				if (it != child_graph.end())
				{
					for (auto &child_id : it->second)
					{
						// Unpack the ID and access the event directly from storage.
						auto [parent_thread_id, parent_event_id] = unpack_unique_event_id(simple_parent_event.unique_id);
						auto [child_thread_id, child_event_id] = unpack_unique_event_id(child_id);

						// event_id is 1-based, vector is 0-based.
						const auto &parent_event = all_events_storage[parent_thread_id][parent_event_id - 1];
						const auto &child_event = all_events_storage[child_thread_id][child_event_id - 1];

						if (child_event.filename == parent_event.filename &&
							child_event.function == parent_event.function &&
							child_event.line == parent_event.line)
							continue;

						child_events.push_back(&child_event);
					}
				}
			}

			return create_simple_events(child_events);
		};

		class EventGroup
		{
		public:
			void calculateStats(unsigned int non_center_percent,
								const std::deque<t_events> &all_events_storage,
								const std::unordered_map<int_fast64_t, std::vector<int_fast64_t>> &child_graph)
			{
				if (all_events.size() == 0)
					return;

				auto all_events_simple = create_simple_events(all_events);
				std::sort(OPT_EXEC_POLICY all_events_simple.begin(), all_events_simple.end(), cmp_simple_event_by_duration_asc);
				all_cnt = static_cast<unsigned int>(all_events_simple.size());
				const double factor = (1.0 / static_cast<double>(all_cnt));

				auto all_child_events_simple = load_child_events_simple(all_events_simple, all_events_storage, child_graph);

				all_time_acc = sum_field(all_events_simple, &Simple_Event::duration);

				const double all_mean = all_time_acc * factor;
				if (std::fpclassify(all_mean) == FP_ZERO)
					return;

				all_st = calculate_std_dev_field(all_events_simple, &Simple_Event::duration, all_mean); // std::sqrt(all_variance);
				all_cv = all_st / all_mean;

				all_thread_cnt = static_cast<unsigned int>(get_distinct_field_values(all_events, &Event::thread_id).size());
				unsigned int amount_non_center = all_cnt * non_center_percent / 100;

				fastest_range = non_center_percent;
				slowest_range = 100 - non_center_percent;

				std::vector<Simple_Event> fastest_events_simple, slowest_events_simple, center_events_simple;
				fastest_events_simple.reserve(amount_non_center);
				slowest_events_simple.reserve(amount_non_center);
				if (all_cnt > 2 * amount_non_center) // ensure no underflow
					center_events_simple.reserve(all_cnt - 2 * amount_non_center);

				for (unsigned int i = 0; i < all_events_simple.size(); i++)
				{
					if (i < amount_non_center)
					{
						fastest_events_simple.push_back(all_events_simple[i]);
					}
					else if (i >= all_cnt - amount_non_center)
					{
						slowest_events_simple.push_back(all_events_simple[i]);
					}
					else
					{
						center_events_simple.push_back(all_events_simple[i]);
					}
				}
				if (!fastest_events_simple.empty())
				{
					fastest_min = fastest_events_simple.front().duration;
					fastest_mean = sum_field(fastest_events_simple, &Simple_Event::duration) / static_cast<double>(fastest_events_simple.size());
				}
				if (!slowest_events_simple.empty())
				{
					slowest_max = slowest_events_simple.back().duration;
					slowest_mean = sum_field(slowest_events_simple, &Simple_Event::duration) / static_cast<double>(slowest_events_simple.size());
				}

				if (!center_events_simple.empty())
				{
					center_min = center_events_simple.front().duration;
					center_max = center_events_simple.back().duration;
					center_mean = sum_field(center_events_simple, &Simple_Event::duration) / static_cast<double>(center_events_simple.size());
					if (center_events_simple.size() % 2 == 1)
						center_med = center_events_simple[center_events_simple.size() / 2].duration;
					else
						center_med = (center_events_simple[center_events_simple.size() / 2].duration + center_events_simple[center_events_simple.size() / 2 - 1].duration) / 2;

					auto center_child_events_simple = load_child_events_simple(center_events_simple, all_events_storage, child_graph);

					std::sort(OPT_EXEC_POLICY center_events_simple.begin(), center_events_simple.end(), cmp_simple_event_by_start_time_asc);
					center_grouped = sorted_create_grouped_simple_events(center_events_simple);
					center_time_active = sum_field(center_grouped, &Simple_Event::duration);

					std::sort(OPT_EXEC_POLICY center_child_events_simple.begin(), center_child_events_simple.end(), cmp_simple_event_by_start_time_asc);
					auto center_child_events_grouped = sorted_create_grouped_simple_events(center_child_events_simple);
					center_time_active_exclusive = center_time_active - sum_field(center_child_events_grouped, &Simple_Event::duration);
				}

				std::sort(OPT_EXEC_POLICY all_events_simple.begin(), all_events_simple.end(), cmp_simple_event_by_start_time_asc);
				all_grouped = sorted_create_grouped_simple_events(all_events_simple);
				all_time_active = sum_field(all_grouped, &Simple_Event::duration);

				std::sort(OPT_EXEC_POLICY all_child_events_simple.begin(), all_child_events_simple.end(), cmp_simple_event_by_start_time_asc);
				auto all_child_events_grouped = sorted_create_grouped_simple_events(all_child_events_simple);
				all_time_active_exclusive = all_time_active - sum_field(all_child_events_grouped, &Simple_Event::duration);
			}

			// all_group
			double all_cv = 0.0;
			double all_st = 0.0;
			unsigned int all_cnt = 0;
			uint_fast64_t all_time_acc = 0;
			uint_fast64_t all_time_active = 0;
			uint_fast64_t all_time_active_exclusive = 0;
			unsigned int all_thread_cnt = 0;
			std::vector<Simple_Event> all_grouped = {};
			std::vector<const Event *> all_events = {};
			// fastest_group
			unsigned int fastest_range = 0;
			uint_fast64_t fastest_min = 0;
			double fastest_mean = 0.0;
			// slowest group
			unsigned int slowest_range = 0;
			uint_fast64_t slowest_max = 0;
			double slowest_mean = 0.0;
			// center group
			uint_fast64_t center_min = 0;
			uint_fast64_t center_max = 0;
			uint_fast64_t center_med = 0;
			double center_mean = 0;
			uint_fast64_t center_time_active = 0;
			uint_fast64_t center_time_active_exclusive = 0;
			std::vector<Simple_Event> center_grouped = {};
			std::string_view filename = {};
			std::string_view function_name = {};
			int line = 0;
		};

		typedef std::vector<std::vector<unsigned int>> sub_events;

		struct store
		{
			inline static std::atomic<bool> write_events_locked = false;
			inline static std::mutex event_mutex;
			inline static std::chrono::high_resolution_clock::time_point track_start_time = std::chrono::high_resolution_clock::now();
			inline static std::atomic<unsigned int> store_clear_cnt = 0;

			inline static std::atomic<int> thread_cnt = -1;
			inline static std::deque<t_events> a_events{};
			inline static std::deque<sub_events> a_sub_events{};

			inline static std::deque<unsigned int> a_current_event_id{}, a_current_event_cnt{};
			inline static std::deque<int> a_thread_ids{};
		};

		inline thread_local t_events *event_ptr = nullptr;
		inline thread_local sub_events *sub_events_ptr = nullptr;
		inline thread_local unsigned int *current_event_id = nullptr;
		inline thread_local unsigned int *current_event_cnt = nullptr;
		inline thread_local int *thread_id = nullptr;

		struct EventKey
		{
			std::string_view filename;
			std::string_view function;
			int line;

			bool operator==(const EventKey &other) const
			{
				return line == other.line && filename == other.filename && function == other.function;
			}
		};

		// A hash function for EventKey so it can be used in `std::unordered_map`.
		struct EventKeyHash
		{
			std::size_t operator()(const EventKey &k) const
			{
				// A simple hash combination.
				std::size_t h1 = std::hash<std::string_view>{}(k.filename);
				std::size_t h2 = std::hash<std::string_view>{}(k.function);
				std::size_t h3 = std::hash<int>{}(k.line);
				// Combine hashes
				return h1 ^ (h2 << 1) ^ (h3 << 2);
			}
		};

		typedef std::unordered_map<EventKey, EventGroup, EventKeyHash> event_group_map;

		struct ctrack_result_settings
		{
			unsigned int non_center_percent = 1;
			double min_percent_active_exclusive = 0.0;			   // between 0-100
			double percent_exclude_fastest_active_exclusive = 0.0; // between 0-100
		};

		class ctrack_result
		{
		public:
			ctrack_result(const ctrack_result_settings &settings, const std::chrono::high_resolution_clock::time_point &track_start_time, const std::chrono::high_resolution_clock::time_point &track_end_time) : settings(settings), track_start_time(track_start_time), track_end_time(track_end_time)
			{
				time_total = std::chrono::duration_cast<std::chrono::nanoseconds>(
								 track_end_time - track_start_time)
								 .count();
				center_intervall_str = "[" + std::to_string(settings.non_center_percent) + "-" + std::to_string(100 - settings.non_center_percent) + "]";
			}

			template <typename StreamType>
			void get_summary_table(StreamType &stream, bool use_color = false)
			{
				BeautifulTable info({
										"Start",
										"End",
										"time total",
										"time ctracked",
										"time ctracked %",
									},
									use_color, alternate_colors);
				info.addRow({BeautifulTable::table_timepoint(track_start_time), BeautifulTable::table_timepoint(track_end_time),
							 BeautifulTable::table_time(time_total), BeautifulTable::table_time(sum_time_active_exclusive),
							 BeautifulTable::table_percentage(sum_time_active_exclusive, time_total)});

				info.print(stream);
				BeautifulTable table({"filename", "function", "line", "calls", "ae" + center_intervall_str + "%", "ae[0-100]%",
									  "time ae[0-100]", "time a[0-100]"},
									 use_color, alternate_colors);
				for (auto &entry : sorted_events)
				{
					table.addRow({BeautifulTable::stable_shortenPath(std::string(entry->filename)), std::string(entry->function_name), BeautifulTable::table_string(entry->line),
								  BeautifulTable::table_string(entry->all_cnt),
								  BeautifulTable::table_percentage(entry->center_time_active_exclusive, time_total),
								  BeautifulTable::table_percentage(entry->all_time_active_exclusive, time_total),
								  BeautifulTable::table_time(entry->all_time_active_exclusive),
								  BeautifulTable::table_time(entry->all_time_active)});
				}

				table.print(stream);
			}
			template <typename StreamType>
			void get_detail_table(StreamType &stream, bool use_color = false, bool reverse_vector = false)
			{
				if (reverse_vector)
				{
					std::reverse(sorted_events.begin(), sorted_events.end());
				}
				for (int i = static_cast<int>(sorted_events.size()) - 1; i >= 0; i--)
				{
					auto &entry = sorted_events[i];

					BeautifulTable info({"filename", "function", "line", "time acc", "sd", "cv", "calls", "threads"}, use_color, default_colors);
					info.addRow({BeautifulTable::stable_shortenPath(std::string(entry->filename)), std::string(entry->function_name), BeautifulTable::table_string(entry->line),
								 BeautifulTable::table_time(entry->all_time_acc),
								 BeautifulTable::table_time(sorted_events[i]->all_st), BeautifulTable::table_string(sorted_events[i]->all_cv),
								 BeautifulTable::table_string(sorted_events[i]->all_cnt), BeautifulTable::table_string(sorted_events[i]->all_thread_cnt)});

					BeautifulTable table({"min", "mean", "min", "mean", "med", "time a", "time ae", "max", "mean", "max"}, use_color, default_colors,
										 {{"fastest[0-" + std::to_string(settings.non_center_percent) + "]%", 2}, {"center" + center_intervall_str + "%", 6}, {"slowest[" + std::to_string(100 - settings.non_center_percent) + "-100]%", 2}});

					table.addRow({BeautifulTable::table_time(entry->fastest_min), BeautifulTable::table_time(entry->fastest_mean),
								  BeautifulTable::table_time(entry->center_min), BeautifulTable::table_time(entry->center_mean),
								  BeautifulTable::table_time(entry->center_med), BeautifulTable::table_time(entry->center_time_active),
								  BeautifulTable::table_time(entry->center_time_active_exclusive),
								  BeautifulTable::table_time(entry->center_max),
								  BeautifulTable::table_time(entry->slowest_mean), BeautifulTable::table_time(entry->slowest_max)});

					info.print(stream);
					table.print(stream);

					stream << std::endl;
				}
			}

			void calculate_stats()
			{
				std::vector<Simple_Event> grouped_events{};

				for (auto &[key, event_group] : m_event_groups)
				{
					ctracked_uses++;
					event_group.filename = key.filename;
					event_group.function_name = key.function;
					event_group.line = key.line;
					event_group.calculateStats(settings.non_center_percent, m_events_storage, child_graph);
					sorted_events.push_back(&event_group);
					grouped_events.insert(grouped_events.end(), event_group.all_grouped.begin(), event_group.all_grouped.end());
				}
				ctracked_files = get_distinct_field_values(sorted_events, &EventGroup::filename).size();
				ctracked_functions = get_distinct_field_values(sorted_events, &EventGroup::function_name).size();

				std::sort(OPT_EXEC_POLICY grouped_events.begin(), grouped_events.end(), cmp_simple_event_by_start_time_asc);
				auto all_grouped = sorted_create_grouped_simple_events(grouped_events);
				sum_time_active_exclusive = sum_field(all_grouped, &Simple_Event::duration);

				order_pointer_vector_by_field(sorted_events, &EventGroup::all_time_active_exclusive, false);

				int fastest_events = static_cast<int>(sorted_events.size() * settings.percent_exclude_fastest_active_exclusive / 100);
				// remove fastest keep in mind fastest elements are at the back
				if (fastest_events > 0 && static_cast<size_t>(fastest_events) < sorted_events.size())
					sorted_events.erase(sorted_events.end() - fastest_events, sorted_events.end());

				uint_fast64_t min_time_active_exclusive = static_cast<uint_fast64_t>(time_total * settings.min_percent_active_exclusive / 100);
				// remove fastest keep in mind fastest elements are at the back
				if (min_time_active_exclusive > 0)
					sorted_events.erase(std::remove_if(sorted_events.begin(), sorted_events.end(), [min_time_active_exclusive](EventGroup *e)
													   { return e->all_time_active_exclusive < min_time_active_exclusive; }),
										sorted_events.end());
			}

			void move_events_from_store(std::deque<t_events> &events)
			{
				m_events_storage = std::move(events);
			}

			void populate_groups()
			{
				for (const auto &event_vec : m_events_storage)
				{
					for (const auto &event : event_vec)
					{
						EventKey key{event.filename, event.function, event.line};
						m_event_groups[key].all_events.push_back(&event);
					}
				}
			}

			void add_sub_events(const sub_events &s_events, const unsigned int thread_id_)
			{

				for (unsigned int parent_idx = 0; parent_idx < s_events.size(); ++parent_idx)
				{
					if (s_events[parent_idx].empty())
						continue;

					// event_id is 1-based, so parent_idx+1 is the parent's event_id
					int_fast64_t parent_id = get_unique_event_id(thread_id_, parent_idx + 1);
					auto &children = child_graph[parent_id];
					children.reserve(s_events[parent_idx].size());
					for (const auto &child_event_id : s_events[parent_idx])
					{
						children.push_back(get_unique_event_id(thread_id_, child_event_id));
					}
				}
			}

			event_group_map m_event_groups{};

			std::unordered_map<int_fast64_t, std::vector<int_fast64_t>> child_graph{};
			ctrack_result_settings settings;
			std::chrono::high_resolution_clock::time_point track_start_time, track_end_time;
			uint_fast64_t time_total;
			uint_fast64_t sum_time_active_exclusive = 0;

			uint_fast64_t ctracked_files = 0;
			uint_fast64_t ctracked_functions = 0;
			uint_fast64_t ctracked_uses = 0;

			std::vector<EventGroup *> sorted_events{};
			std::string center_intervall_str;

		private:
			std::deque<t_events> m_events_storage;
		};

		inline int fetch_event_t_id()
		{
			if (thread_id == nullptr || *thread_id == -1)
			{
				std::scoped_lock lock(store::event_mutex);

				if (thread_id == nullptr)
				{
					store::a_thread_ids.emplace_back(++store::thread_cnt);
					thread_id = &store::a_thread_ids.back();
				}
				else
				{
					*thread_id = ++store::thread_cnt;
				}

				store::a_events.emplace_back(t_events{});
				store::a_sub_events.emplace_back(sub_events{});
				store::a_current_event_id.emplace_back(0);
				store::a_current_event_cnt.emplace_back(0);

				event_ptr = &store::a_events[*thread_id];
				sub_events_ptr = &store::a_sub_events[*thread_id];

				current_event_id = &store::a_current_event_id[*thread_id];
				current_event_cnt = &store::a_current_event_cnt[*thread_id];

				event_ptr->reserve(100);
			}
			return *thread_id;
		}

		class EventHandler
		{
		public:
			EventHandler(int line = __builtin_LINE(), const char *filename = __builtin_FILE(), const char *function = __builtin_FUNCTION(), std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now()) : line(line)
			{
				previous_store_clear_cnt = store::store_clear_cnt;
				this->filename = filename;
				this->function = function;
				while (store::write_events_locked)
				{
				}

				register_event();
				this->start_time = start_time;
			}
			~EventHandler()
			{
				auto end_time = std::chrono::high_resolution_clock::now();
				while (store::write_events_locked)
				{
				}

				if (store::store_clear_cnt != previous_store_clear_cnt)
				{
					register_event();
				}

				if (event_ptr->capacity() == event_ptr->size())
					event_ptr->reserve(event_ptr->capacity() * 1.8);

				event_ptr->emplace_back(Event{start_time, end_time, filename, line, function, t_id, event_id});

				*current_event_id = previous_event_id;

				if (previous_event_id > 0)
				{

					size_t parent_index = previous_event_id - 1;
					if (sub_events_ptr->size() <= parent_index)
					{
						sub_events_ptr->resize(parent_index + 1);
					}
					(*sub_events_ptr)[parent_index].push_back(event_id);
				}
			}

		private:
			void register_event()
			{
				t_id = fetch_event_t_id();
				previous_event_id = *current_event_id;
				// event_id is 1-based.
				event_id = ++(*current_event_cnt);
				*current_event_id = event_id;
			}
			std::chrono::high_resolution_clock::time_point start_time;
			int line;
			unsigned int previous_store_clear_cnt;
			std::string_view filename, function;
			int t_id;
			unsigned int event_id;
			unsigned int previous_event_id;
		};

		inline void clear_a_store()
		{
			store::a_current_event_id.clear();
			store::a_current_event_id.shrink_to_fit();

			store::a_current_event_cnt.clear();
			store::a_current_event_cnt.shrink_to_fit();

			store::a_events.clear();
			store::a_events.shrink_to_fit();

			store::a_sub_events.clear();
			store::a_sub_events.shrink_to_fit();

			store::thread_cnt = -1;
			for (auto &entry : store::a_thread_ids)
			{
				entry = -1;
			}
			store::store_clear_cnt++;
			store::track_start_time = std::chrono::high_resolution_clock::now();
		}

		inline ctrack_result calc_stats_and_clear(ctrack_result_settings settings = {})
		{
			auto end = std::chrono::high_resolution_clock::now();
			ctrack_result res{settings, store::track_start_time, end};

			{
				store::write_events_locked = true;
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				std::scoped_lock lock(store::event_mutex);

				res.move_events_from_store(store::a_events);

				res.populate_groups();

				for (int thread_id_ = 0; thread_id_ <= store::thread_cnt; thread_id_++)
				{
					auto &t_sub_events = store::a_sub_events[thread_id_];
					res.add_sub_events(t_sub_events, thread_id_);
				}
				clear_a_store();
				store::write_events_locked = false;
			}

			res.calculate_stats();
			store::track_start_time = std::chrono::high_resolution_clock::now();

			return res;
		}

		inline void result_print(ctrack_result_settings settings = {})
		{
			auto res = calc_stats_and_clear(settings);
			std::cout << "Details" << std::endl;
			res.get_detail_table(std::cout, true);
			std::cout << "Summary" << std::endl;
			res.get_summary_table(std::cout, true);
		}

		inline std::string result_as_string(ctrack_result_settings settings = {})
		{
			auto res = calc_stats_and_clear(settings);
			std::stringstream ss;
			ss << "Summary\n";
			res.get_summary_table(ss, false);
			ss << "Details\n";
			res.get_detail_table(ss, false, true);

			return ss.str();
		}
	}
}

#ifndef CTRACK_DISABLE
#define CTRACK_CONCAT_IMPL(x, y) x##y
#define CTRACK_CONCAT(x, y) CTRACK_CONCAT_IMPL(x, y)
#define CTRACK_UNIQUE_NAME(prefix) CTRACK_CONCAT(prefix, __COUNTER__)

#define CTRACK_IMPL \
	ctrack::EventHandler CTRACK_UNIQUE_NAME(ctrack_instance_) { __builtin_LINE(), __builtin_FILE(), __builtin_FUNCTION() }
#define CTRACK_IMPL_NAME(name) \
	ctrack::EventHandler CTRACK_UNIQUE_NAME(ctrack_instance_) { __builtin_LINE(), __builtin_FILE(), name }
#if defined(CTRACK_DISABLE_DEV)
#define CTRACK_PROD CTRACK_IMPL
#define CTRACK_PROD_NAME(name) CTRACK_IMPL_NAME(name)
#define CTRACK_DEV			  // Disabled
#define CTRACK_DEV_NAME(name) // Disabled
#elif defined(CTRACK_DISABLE_PROD)
#define CTRACK_PROD			   // Disabled
#define CTRACK_PROD_NAME(name) // Disabled
#define CTRACK_DEV CTRACK_IMPL
#define CTRACK_DEV_NAME(name) CTRACK_IMPL_NAME(name)
#else
#define CTRACK_PROD CTRACK_IMPL
#define CTRACK_PROD_NAME(name) CTRACK_IMPL_NAME(name)
#define CTRACK_DEV CTRACK_IMPL
#define CTRACK_DEV_NAME(name) CTRACK_IMPL_NAME(name)
#endif

// Alias CTRACK to CTRACK_PROD
#define CTRACK CTRACK_PROD
#define CTRACK_NAME(name) CTRACK_PROD_NAME(name)

#else // CTRACK_DISABLE
#define CTRACK_PROD
#define CTRACK_PROD_NAME(name)
#define CTRACK_DEV
#define CTRACK_DEV_NAME(name)
#define CTRACK
#define CTRACK_NAME(name)
#endif // CTRACK_DISABLE

#endif
